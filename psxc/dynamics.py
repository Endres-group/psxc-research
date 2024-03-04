from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0, i1
from simple_pytree import Pytree, dataclass, static_field


@dataclass
class CellState(Pytree):
    xf: jax.Array
    xu: jax.Array
    phis: jax.Array
    age: float = 0.0

    @classmethod
    def spawn(cls, key, x, length=0.5, num_pods=12, y=0.0):
        theta = jax.random.uniform(key, shape=(), minval=0.0, maxval=2 * jnp.pi)
        x_focal = jnp.array([x, y])
        x_uro = x_focal - length * jnp.array([jnp.cos(theta), jnp.sin(theta)])
        phis = jnp.linspace(0, 2 * jnp.pi, num_pods + 1).at[-1].set(-jnp.pi)
        return cls(xf=x_focal, xu=x_uro, phis=phis, age=0.0)


@dataclass
class Parameters(Pytree):
    gain_magnitude: Union[jax.Array, float] = 1.0
    activation_rate: float = 3.0
    decay_rate: float = 0.3
    exchange_rate: float = 0.5
    cross_inhibition: float = 0.5
    noise_strength: float = 1.5e-3
    actin_pool: float = 1.0
    growthscale: float = static_field(default=1.0)


class LinearProfile(Pytree):
    grad: float
    c0: float

    def __init__(self, grad, value, x=None):
        self.grad = grad
        if x is not None:
            value = value - grad * x
        self.c0 = value

    def concentration(self, x):
        return jnp.maximum(1e-4, self.grad * x[0] + self.c0)

    def snr(self, x):
        return jnp.where(self.concentration(x) > 0, self.grad**2 / self.concentration(x), 0.0)

    def logsnr(self, x):
        return jnp.log10(self.snr(x))


@partial(jax.jit, static_argnames=('t_max', 'dt', 'has_aux'))
def sensing_event(
    params,
    key,
    cell_state,
    profile,
    signal_strength=jnp.ones(12),
    t_max=25.0,
    dt=0.01,
    has_aux=False,
):

    theta = jnp.arctan2(cell_state.xf[1] - cell_state.xu[1], cell_state.xf[0] - cell_state.xu[0])
    length = jnp.linalg.norm(cell_state.xf - cell_state.xu)
    num_pods = len(cell_state.phis) - 1

    c0 = profile.concentration(cell_state.xf)

    phis = (cell_state.phis + theta) % (2 * jnp.pi)# - jnp.pi
    phi_vec = jnp.array([jnp.cos(phis), jnp.sin(phis)]).T

    def ode_step_fn(carry, dW):
        x, y, t, y_avg, done = carry

        y, yu = y[:-1], y[-1]
        #angular_distance = jnp.abs(phis[:-1, None] - phis[None, :-1]) / (jnp.pi)
        #w = 1.0 * jnp.ones((num_pods, num_pods)) - jnp.eye(num_pods)# + 0.2 * (jnp.eye(num_pods, k=1) + jnp.eye(num_pods, k=-1)) + 0.05 * (jnp.eye(num_pods, k=2) + jnp.eye(num_pods, k=-2))
        #w = 1.0 * jnp.exp(-angular_distance) + 0.2
        w = jnp.ones((num_pods, num_pods))
        w = w.at[jnp.diag_indices(num_pods)].set(0.0)
        y_bar = w @ y

        c = jax.vmap(profile.concentration)(x[:-1])

        logistic_fn = lambda x, x0: 1.0 / (1.0 + jnp.exp(-params.activation_rate * (x - x0)))

        gain = signal_strength * params.gain_magnitude * logistic_fn(c, c0)
        decay = params.decay_rate
        exchange = params.exchange_rate
        cross = params.cross_inhibition

        drift = gain * yu - decay * y - cross * y * y_bar + exchange * (y - y_bar)
        diffusivity = params.noise_strength * jnp.sqrt(c)
        dy = drift * dt + diffusivity * dW

        y_new = jnp.clip(y + dy, 0.0, params.actin_pool)
        yu_new = jnp.clip(params.actin_pool - jnp.sum(y_new, keepdims=True), 0.0, params.actin_pool)
        y_new = jnp.concatenate([y_new, yu_new], axis=0)

        y_avg = jnp.roll(y_avg, shift=1, axis=0).at[0].set(y_new)
        w = jnp.mean(y_avg, axis=0)
        w = w / jnp.sum(w)

        x_new = jax.vmap(lambda y, e: cell_state.xf + y * length * e)(w, phi_vec)
        t_new = t + dt
        done = jnp.any(w[:-1] > 0.95)
        return (x_new, y_new, t_new, y_avg, done), (x_new, y_new)

    def step_fn(carry, dW):
        *_, done = carry
        do_nothing = lambda carry, _: (carry, carry[:2])
        carry, (x, y) = jax.lax.cond(done, do_nothing, ode_step_fn, carry, dW)
        return carry, (x, y, done)

    max_timesteps = int(t_max / dt)

    y_init = jnp.zeros((len(cell_state.phis))).at[-1].set(params.actin_pool)
    dWs = jnp.sqrt(dt) * jax.random.normal(key, (max_timesteps, num_pods))
    x_init = jnp.full((num_pods + 1, 2), fill_value=cell_state.xf).at[-1].set(cell_state.xu)
    y_avg = jnp.zeros((int(params.growthscale // dt), len(cell_state.phis)))

    init = (x_init, y_init, 0.0, y_avg, False)
    (_, y_last, duration, _, _), (xs, ys, dones) = jax.lax.scan(step_fn, init, dWs)

    # Check if the winner it is done, otherwise choose the winner based on y_last
    winner_idx = jax.lax.select(dones[-1], jnp.argmax(y_last[:-1]), jax.random.choice(key, num_pods, p=y_last[:-1]))
    new_head = cell_state.xf + length * phi_vec[winner_idx]
    xs = jnp.insert(xs, -1, cell_state.xf, axis=1)
    if has_aux:
        return (xs, ~dones), new_head, duration, ys
    return (xs, ~dones), new_head, duration


def _default_signal_fn(snr):
    return jnp.ones(12)

@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def run_chemotactic_trajectory(params, key, cell_state, profile, signal_fn=_default_signal_fn,
                               num_steps=20, t_max=15.0, dt=0.01, warmup=5):
    def step_fn(carry, key):
        cell_state = carry
        signal = signal_fn(profile.logsnr(cell_state.xf))
        (xs, mask), xf, _ = sensing_event(params, key, cell_state, profile, signal, t_max, dt)
        cell_state = cell_state.replace(xf=xf, xu=cell_state.xf)
        return cell_state, (xs, mask)

    keys = jax.random.split(key, num_steps)
    _, (xs, mask) = jax.lax.scan(step_fn, cell_state, keys)
    return xs[warmup:], mask[warmup:]


@jax.jit
@jax.vmap
def weighted_chemotactic_index(points):
    diff = points - points[-2, None]
    diff = jnp.delete(diff, -2, axis=0, assume_unique_indices=True)
    lengths = jnp.linalg.norm(diff, axis=-1)
    thetas = jnp.arctan2(diff[..., 1], diff[..., 0])
    thetas = thetas.at[..., -1].add(jnp.pi)
    ci = jnp.cos(thetas)
    w_ci = jnp.average(ci, axis=-1, weights=lengths)
    return w_ci


@jax.jit
def signal_to_noise(xs, profile):
    snr = jax.vmap(profile.logsnr)(xs[:, -2])
    return snr


@jax.jit
def decision_time(xs, mask, *, alpha=0.05, dt=0.01):
    time = jnp.sum(mask, axis=-1) * dt
    lengths = jnp.linalg.norm(xs - xs[:, -2, jnp.newaxis], axis=-1)

    # Set the threshold at which we decide the cell has made a decision
    threshold = alpha * jnp.max(lengths)

    # Remove the length of the uropod and the focal adhesion (which is 0 always)
    L = lengths[:, :-2]

    # Compute the complementary length (sum of all the OTHER pseudopods)
    L_comp = L - (jnp.sum(L, axis=1, keepdims=True) - L)

    # Find the index of the pseudopod that won
    winner_idx = jnp.argmax(L[-1])

    # Find the first time the winner pseudopod has a length greater than the threshold
    # and it continues to be greater than the threshold until the end of the event.
    first_occurance = (
        len(xs) - jnp.argmin(jnp.cumprod((L_comp[:, winner_idx] >= threshold)[::-1])) - 1
    )
    decision_time = first_occurance * dt
    return decision_time, time - decision_time


def put_on_bins(x, y, bins):
    x = np.array(x)
    y = np.array(y)
    assignment = np.digitize(x, bins)
    indices = np.argwhere(np.bincount(assignment, minlength=len(bins)) > 0)[..., 0]
    bins = bins[indices - 1]
    yerr = np.array(
        [np.std(y[assignment == i]) / np.sqrt(np.sum(assignment == i)) for i in indices]
    )
    y = np.array([np.mean(y[assignment == i]) for i in indices])
    return bins, y, yerr


def perfect_absorber_limit(snr, a):
    y = 3 * np.pi * snr * a
    return np.sqrt(np.pi * y / 2) * np.exp(-y) * (i0(y) + i1(y))


def theorical_limit_similarity(grad, value, a, T):
    y = (3 * np.pi * grad**2 * a * T) / value
    similarity = np.sqrt(np.pi * y / 2) * np.exp(-y) * (i0(y) + i1(y))
    return similarity


def cosine_similarity(xf, xu):
    return jnp.cos(jnp.arctan2(xf[1] - xu[1], xf[0] - xu[0]))
