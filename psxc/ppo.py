from functools import partial
from typing import NamedTuple

import flax.linen as nn
from flax.training.train_state import TrainState
from flax.struct import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import optax
from simple_pytree import Pytree
from tqdm import tqdm, trange

import psxc


class ExperimentConfig(NamedTuple):
    n_train_envs: int = int(2**12)  # Number of train environments in parallel.
    total_steps: int = int(3e8)  # Number of train steps.
    n_eval_envs: int = int(2**10)  # Number of parallel envs to evaluate on.
    learning_rate: float = 3e-4  # Learning rate.
    n_epochs: int = 8  # Number of epochs at each training step.
    n_minibatch: int = 8  # Number of minibatches to split the buffer into.
    clip_eps: float = 0.1  # Surrogate clipping loss.
    critic_coeff: float = 0.5  # Value loss coefficient
    discount: float = 0.99  # Discount factor for the GAE calculation.
    gae_lambda: float = 0.99  # "GAE lambda"
    logdir: str = "./runs/"  # Path to store the logs.
    seed: int = 42  # Random state seed.
    n_steps: int = 30  # Number of steps per environment before training.
    dt: float = 0.1  # Time step for the simulation.
    discrete: bool = True  # Discrete action space.
    length: float = 1.0  # Length of the cell.


class Rollout(NamedTuple):
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    log_probs: jax.Array
    values: jax.Array
    mask: jax.Array


@dataclass
class EnvState:
    cell_state: psxc.CellState
    step: int
    profile: psxc.LinearProfile
    x0: float
    done: bool


class CellEnv(Pytree):
    """Environment of a chemotactic cell where at each step it does a splitting event."""

    observation_space: int = 1

    def __init__(self, env_params, n_envs, x0=50, dt=0.1, t_max=50.0, max_steps=10, penalty=0.5):
        """Initialize the environment.
        Args:
            env_params: The parameters of the environment.
            n_envs: Number of environments to run in parallel.
            x0: Initial position of the cell.
            dt: Time step for the simulation.
            t_max: Maximum time for a splitting event.
            max_steps: Maximum number of steps before the environment is done.
            penalty: Penalty for the time regularization term.
        """
        self.env_params = env_params
        self.n_envs = n_envs
        self.x0 = x0
        self.max_steps = max_steps - 2
        self.length = 1.0
        self.t_max = t_max
        self.dt = dt
        self.penalty = penalty

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        key_pos, key_init = jax.random.split(key, 2)
        xs = jax.random.normal(key_pos, (self.n_envs,)) * 10 + self.x0
        x0s = jnp.ones_like(xs) * self.x0
        keys = jax.random.split(key_init, self.n_envs)

        def init_env_state(key, x, x0):
            """Initialize the environment state."""
            cell_state = psxc.CellState.spawn(key, x=x, length=1.0)
            key_grad, key_val = jax.random.split(key)
            grad = 10 ** jax.random.uniform(key_grad, (), minval=-3, maxval=0.4)
            value = jax.random.uniform(key_val, (), minval=25.0, maxval=175.0)
            profile = psxc.LinearProfile(grad, value, x=x0)
            obs = self._get_obs(cell_state, profile)
            state = EnvState(cell_state, 0, profile, x0, False)
            return obs, state

        return jax.vmap(init_env_state)(keys, xs, x0s)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, env_states, actions):
        """Step the environment forward given the actions."""

        def _step(env_state, action):
            cell_state = env_state.cell_state

            # Perform the sensing event and get the new state.
            # The action is the signal strength for each of the pseudopods.
            _, xf, T = psxc.sensing_event(self.env_params, key, cell_state, env_state.profile, action, t_max=self.t_max, dt=self.dt)
            obs = self._get_obs(cell_state, env_state.profile)

            # Compute the reward as the orientation reward and a time regularization term.
            # Only start computing the reward after the first step.
            orientation_reward = (xf[0] - cell_state.xf[0]) / self.length
            time_regularization = self.penalty * (self.t_max - T) / self.t_max
            reward = jnp.where(env_state.step <= 2, 0.0, orientation_reward + time_regularization)

            # Update the cell state and the environment state.
            cell_state = cell_state.replace(xf=xf, xu=cell_state.xf, age=cell_state.age + T)
            done = env_state.step >= self.max_steps
            env_state = EnvState(cell_state, env_state.step + 1, env_state.profile, env_state.x0, done)
            return obs, env_state, reward, done, {}

        def _do_nothing(env_state, _):
            # Do nothing if the environment is done.
            obs_re = jnp.array([0.0])
            return obs_re, env_state, 0.0, True, {}

        @jax.vmap
        def step(env_state, action):
            return jax.lax.cond(env_state.done, _do_nothing, _step, env_state, action)

        actions = jnp.clip(actions, 0.0, 1.0)
        return step(env_states, actions)

    def _get_obs(self, cell_state, profile):
        """Get the observation of the environment."""
        # The observation is the log10 of the signal to noise ratio.
        snr = profile.logsnr(cell_state.xf)
        return jnp.array([snr])


class ActorCritic(nn.Module):
    outdim: int = 12
    discrete: bool = True

    @nn.compact
    def __call__(self, x):
        init_fn = nn.initializers.orthogonal(scale=jnp.sqrt(2))
        init_fn_actor = nn.initializers.orthogonal(scale=0.1)

        # Critic network
        x_v = x
        for _ in range(4):
            x_v = nn.tanh(nn.Dense(128, kernel_init=init_fn)(x_v))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal())(x_v)

        # Actor Network
        x_a = x
        for _ in range(4):
            x_a = nn.tanh(nn.Dense(128, kernel_init=init_fn_actor)(x_a))

        if self.discrete:  # Discrete actin space [0 or 1]^n
            logits = nn.Dense(self.outdim * 2)(x_a)
            logits = logits.reshape((*logits.shape[:-1], self.outdim, 2))
            return value, logits

        # Continuous action space (0, 1)^n
        mu = nn.sigmoid(nn.Dense(self.outdim, kernel_init=init_fn)(x_a))
        log_scale = nn.Dense(self.outdim, kernel_init=init_fn)(x_a)
        scale = jax.lax.clamp(0.05, jax.nn.softplus(-0.5 + log_scale), 0.5)
        return value, (mu, scale)


def loss_fn(params, apply_fn, minibatch, eps=0.1, entropy_coeff=1e-6, vf_coeff=0.5, discrete=True):
    s, a, logp_old, target, A, mask = minibatch

    # Sample the actions from the policy and compute the log probabilities.
    if discrete:
        values, logits = apply_fn(params, s)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        logits_taken = jnp.take_along_axis(log_probs, a[..., None], axis=-1)[..., 0]
        logp = logits_taken.sum(-1)
        entropy = -jnp.sum(jax.nn.log_softmax(logits) * jax.nn.softmax(logits), axis=-1).mean(-1)
    else:
        values, (mu, scale) = apply_fn(params, s[..., None])
        mu = jnp.swapaxes(mu, -1, -2)[..., 0]
        scale = jnp.swapaxes(scale, -1, -2)[..., 0]
        logp = jax.scipy.stats.norm.logpdf(a, loc=mu, scale=scale).sum(-1)

        # Entropy of normal distribution is H = -p*ln(p) = 1/2 ln(e*2π*σ^2)
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(scale), axis=-1)

    # Normalize the advantage and compute the policy loss.
    A = (A - jnp.mean(A, where=mask)) / (jnp.std(A, where=mask) + 1e-8)
    ratio = jnp.exp(logp - logp_old)
    policy_loss = jnp.minimum(ratio * A, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * A)
    policy_loss = -jnp.mean(policy_loss, where=mask)

    # Compute critic  and entropy loss
    values = values[..., 0]
    value_loss = jnp.mean((values - target) ** 2, where=mask)
    entropy_loss = jnp.mean(entropy, where=mask)

    # Compute KL divergence (approximation)
    approx_kl_div = jnp.mean((ratio - 1) - (logp - logp_old), where=mask)

    loss = policy_loss + vf_coeff * value_loss - entropy_coeff * entropy_loss
    aux = (loss, policy_loss, value_loss, entropy_loss, approx_kl_div)
    return loss, aux


@partial(jax.jit, static_argnums=(3,))
def train_step(key, train_state, batch, config):
    """Perform a training step with the PPO algorithm."""

    buffer_size = batch[0].shape[1]
    grad_fn = jax.grad(loss_fn, has_aux=True)

    def epoch_step(state, key):
        """Perform n_minibatch*n_epochs gradient descent steps."""

        def batch_step(state, chosen):
            minibatch = jax.tree.map(lambda x: x[:, chosen], batch)
            grads, metrics = grad_fn(state.params, state.apply_fn, minibatch, discrete=config.discrete)
            state = state.apply_gradients(grads=grads)
            return state, metrics

        batch_indices = jax.random.permutation(key, buffer_size)
        batch_indices = batch_indices.reshape(config.n_minibatch, -1)
        state, metrics = jax.lax.scan(batch_step, init=state, xs=batch_indices)
        return state, jax.tree.map(jnp.mean, metrics)

    keys = jax.random.split(key, config.n_epochs)
    train_state, metrics = jax.lax.scan(epoch_step, init=train_state, xs=keys)
    return train_state, jax.tree.map(jnp.mean, metrics)


def calculate_gae(values, rewards, dones, discount=0.99, A_lambda=0.95):
    """Calculate the Generalized Advantage Estimation."""

    def body_fn(A, x):
        next_value, done, value, reward = x
        value_diff = discount * next_value * (1 - done) - value
        delta = reward + value_diff
        A = delta + discount * A_lambda * (1 - done) * A
        return A, A

    xs = (values[1:], dones[:-1], values[:-1], rewards[:-1])
    num_envs = values.shape[1]
    _, gae = jax.lax.scan(body_fn, jnp.zeros(num_envs), xs, reverse=True)
    gae = jnp.pad(gae, pad_width=((0, 1), (0, 0)))
    return gae


@jax.jit
def collect_batch(buffer, discount=1.0, A_lambda=0.95):
    """Collect the batch from the buffer and compute the GAE."""
    gae = calculate_gae(buffer.values, buffer.rewards, buffer.dones, discount, A_lambda)
    target = gae + buffer.values
    batch = (buffer.states, buffer.actions, buffer.log_probs, target, gae, buffer.mask)

    # Remove the last element of the batch since it doesn't have advantage to compute.
    batch = jax.tree.map(lambda x: x[:-1], batch)
    batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
    batch = jax.tree.map(lambda x: x[None, ...], batch)
    return batch


def init_train_state(key, env, config):
    """Initialize the training state with the model and the optimizer."""
    model = ActorCritic(12, discrete=config.discrete)
    dummy_x = jnp.ones((1, 1, env.observation_space))
    params = model.init(key, dummy_x)

    tx = optax.adam(config.learning_rate, eps=1e-7)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=(2, 3))
def evaluate(state, env_params, config, deterministic=False):
    """Deterministic evaluation of the policy."""
    eval_envs = CellEnv(env_params, config.n_eval_envs, config.dt, max_steps=config.n_steps)
    eval_key = jax.random.key(42)

    def transition_step(carry, key):
        s, env_state = carry
        key_action, key_step = jax.random.split(key, 2)
        if config.discrete:
            _, logits = state.apply_fn(state.params, s)
            a = jax.random.categorical(key_action, logits, axis=-1)
            a = jax.lax.select(deterministic, jnp.argmax(jax.nn.softmax(logits), -1), a)
        else:
            _, (mu, scale) = state.apply_fn(state.params, s)
            a = jax.random.normal(key_action, shape=mu.shape) * scale + mu
            a = jax.lax.select(deterministic, mu, a)
        next_s, next_env_state, reward, done, __ = eval_envs.step(key_step, env_state, a)
        return (next_s, next_env_state), (reward, done)

    obs, env_state = eval_envs.reset(eval_key)
    keys = jax.random.split(eval_key, config.n_steps)
    _, (rewards, dones) = jax.lax.scan(transition_step, (obs, env_state), keys)

    reward = jnp.sum(rewards * (1 - dones), axis=0)
    return jnp.mean(reward)


def train_loop(key, config, env_params, checkpointer):
    """Main training loop for the PPO algorithm."""

    envs = CellEnv(env_params, config.n_train_envs, config.dt, max_steps=config.n_steps)

    key, key_init = jax.random.split(key, num=2)
    train_state = init_train_state(key_init, envs, config)

    @jax.jit
    def run_episodes(train_state, key):
        """Run n_steps in parallel for each of the num_train_envs."""

        def step(carry, key):
            train_state, obs, env_state, mask = carry
            key_action, key_step = jax.random.split(key, 2)

            if config.discrete:
                value, logits = train_state.apply_fn(train_state.params, obs)
                actions = jax.random.categorical(key_action, logits, axis=-1)
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                logits_taken = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
                log_prob = logits_taken.sum(-1)

            else:
                value, (mu, scale) = train_state.apply_fn(train_state.params, obs)
                actions = jax.random.normal(key_action, shape=mu.shape) * scale + mu
                log_prob = jax.scipy.stats.norm.logpdf(actions, mu, scale).sum(-1)

            # Step the environment and collect the rollout.
            next_obs, next_env_state, reward, done, _ = envs.step(key_step, env_state, actions)
            rollout = Rollout(obs, actions, reward, done, log_prob, value[..., 0], 1 - mask)

            return (train_state, next_obs, next_env_state, mask | done), rollout

        key, key_reset = jax.random.split(key, 2)
        keys = jax.random.split(key, config.n_steps + 1)

        obs, env_state = envs.reset(key_reset)
        mask = jnp.zeros(config.n_train_envs) > 1
        init = (train_state, obs, env_state, mask)
        _, buffer = jax.lax.scan(step, init, keys)
        return buffer

    # Divide the total steps by the step size to get the number of steps to run.
    step_size = int(config.n_train_envs * config.n_steps)
    total_steps = np.ceil(config.total_steps // step_size).astype(int)
    eval_interval = max(1, int(total_steps / 100))

    for step in (pbar := trange(total_steps, ncols=81, total=config.total_steps)):

        # Perform n_steps for each of the num_train_envs in parallel.
        buffer = run_episodes(train_state, jax.random.fold_in(key, step))
        pbar.update(step_size)

        # Collect the buffer from the rollout and perform a training step corresponding
        # to n_minibatch*n_epochs gradient descent steps with the current policy.
        key_train = jax.random.fold_in(key, step)
        batch = collect_batch(buffer, config.discount, config.gae_lambda)
        train_state, metrics = train_step(key_train, train_state, batch, config)

        # Logkeeping the metrics and saving the model.
        checkpointer.writer.add_scalar("train/loss", metrics[0], pbar.n)
        checkpointer.writer.add_scalar("train/policy_loss", metrics[1], pbar.n)
        checkpointer.writer.add_scalar("train/critic_loss", metrics[2], pbar.n)
        checkpointer.writer.add_scalar("train/entropy_loss", metrics[3], pbar.n)
        checkpointer.writer.add_scalar("train/kl_divergence", metrics[4], pbar.n)

        # Evaluate the model every eval_interval steps.
        if step % eval_interval == 0:
            eval_reward = evaluate(train_state, env_params, config, deterministic=False)
            checkpointer.writer.add_scalar("eval/reward", eval_reward, pbar.n)
            pbar.set_description(f"R: {eval_reward:.2g}")

            checkpointer.manager.save(checkpointer.dir.absolute() / "checkpoint", train_state.params, force=True)

    return train_state
