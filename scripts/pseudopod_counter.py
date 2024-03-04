from functools import partial
import itertools
import numpy as np
import matplotlib.pyplot as plt
import jax
import pathlib
import jax.numpy as jnp

import psxc

# For different gradients
grads = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
vals = [75.0]
experiments = list(itertools.product(grads, vals))

x0 = 50.0

params = psxc.Parameters(activation_rate=1.0)


@partial(jax.jit, static_argnames=("nreps", "dt"))
def evaluate(key, grad, val, params, nreps=10, dt=0.1):

    profile = psxc.LinearProfile(grad, val, x=x0)

    def find_lengths(xs, mask, *, alpha=0.1):
        duration = jnp.sum(mask, axis=-1) * dt
        lengths = jnp.linalg.norm(xs - xs[:, -2, jnp.newaxis], axis=-1)
        threshold = alpha * jnp.max(lengths)
        L = lengths[:, :-2]
        return L

    def calc_decision(key):
        key_init, key_sim = jax.random.split(key, num=2)
        cell_state = psxc.CellState.spawn(key_init, x=x0)
        (xs, mask), xf, T = psxc.sensing_event(params, key_sim, cell_state, profile, t_max=20.0, dt=dt)
        lengths = find_lengths(xs, mask)
        return {"acc": xf[0] > cell_state.xf[0], "lengths": lengths}

    keys = jax.random.split(key, nreps)
    return jax.vmap(calc_decision)(keys)

key = jax.random.key(42)
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12,12))
for ax, (grad, val) in zip(axes.flatten(), experiments):
    results = evaluate(key, grad, val, params)
    L = results['lengths'][1]
    ax.set_title(f"Grad {grad}, Value {val}")
    for i in range(L.shape[-1]):
        ax.plot(L[:,i])
plt.tight_layout()
plt.show()
