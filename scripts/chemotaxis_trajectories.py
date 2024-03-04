"""
This script generates a figure showing chemotactic trajectories of cells in a linear gradient.
The figure shows the trajectories of cells in a linear gradient of different slopes.
The trajectories are color-coded by the gradient of the linear gradient.
The figure also shows the distribution of final position of the cells in the gradient.

The figure is saved in the `figures` directory.

Author: Albert Alonso
Date: 2024.03.25
"""
import pathlib
from collections import namedtuple

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # pyright: ignore
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.collections import LineCollection

import psxc

matplotlib.use("Agg")

# Parameters
SEED = 42
n_trajectories = 300

grad = 0.5
value = 75
x0 = 50

Trajectory = namedtuple("Trajectory", ["x", "mask"])

current_dir = pathlib.Path(__file__).parent

def run(key, params, dt=0.05):

    def calculate_com(xs):
        # xs shape: (num_steps, duration_step, num_points, 2) -> (num_steps, 2)
        return jnp.mean(xs, axis=2)

    def collapse(x):
        return x.reshape(-1, *x.shape[2:])

    def run_single(key):
        key_init, key_sim = jax.random.split(key)
        cell_state = psxc.CellState.spawn(key_init, x=x0, length=1.0)
        xs, mask = psxc.run_chemotactic_trajectory(
            params, key_sim, cell_state, profile, num_steps=100, warmup=0, t_max=30, dt=dt
        )
        center_of_mass = calculate_com(xs)
        return jax.tree_util.tree_map(collapse, Trajectory(center_of_mass, mask))

    keys = jax.random.split(key, n_trajectories)
    return jax.vmap(run_single)(keys)


key = jax.random.key(SEED)
params = psxc.Parameters()

results = {}
for grad in [0.01, 0.2, 2.0]:
    profile = psxc.LinearProfile(grad, value, x=x0)
    trajectories = run(key, params, dt=0.01)
    results[grad] = jax.tree.map(np.asarray, trajectories)

# Plot
figures_dir = current_dir / "figures"
figures_dir.mkdir(exist_ok=True)

with plt.style.context(['science']):

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.5, 3.5), gridspec_kw={'height_ratios': [1.2, 1]}, sharex=True)

    profile = psxc.LinearProfile(0.2, value, x=x0)

    pad = 1.0
    xmin = 20
    xmax = 140
    ymin=-20
    ymax=20
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)
    c = jax.vmap(jax.vmap(profile.concentration))(np.dstack([X, Y]))


    ax.pcolormesh(x, y, c, cmap='Blues', alpha=0.5, lw=0, rasterized=True)
    ax.axvline(x0, color='black', linestyle='-')

    #scalebar = ScaleBar(5, location='lower right', units='um', box_color='none')
    #ax.add_artist(scalebar)

    for i in range(10):
        n_traj_batch = n_trajectories // 10
        slices = np.arange(i*n_traj_batch, (i+1)*n_traj_batch)
        for cmap, grad in zip(['Grays','Greens', 'Reds'], results):
            color = plt.colormaps[cmap](np.random.rand(n_traj_batch)*0.8+0.4)
            trajectories = results[grad]
            for i, (x, mask) in enumerate(zip(trajectories.x[slices], trajectories.mask[slices])):
                # Stop at 10 minutes (10 * 60 * 100)
                x = x[mask][:60_000]
                collection = LineCollection([x], color=color[i], alpha=0.5, lw=0.5, rasterized=True)
                ax.add_collection(collection)
                profile = psxc.LinearProfile(grad, value, x=x0)

            _xmin = np.array([trajectories.x[...,0].min(), 0.0])
            _xmax = np.array([trajectories.x[...,0].max(), 0.0])
            print(f'SNR for {grad}:', profile.logsnr(_xmin), profile.logsnr(_xmax))

    ax.set_aspect("equal")
    ax.set(yticklabels=[])


    # Plot histogram of x values
    bins = np.linspace(xmin, xmax, 100)
    bx.axvline(x0, color='black', linestyle='-')
    for color, grad, label in zip(['gray', 'green', 'red'], results, ['low', 'mid', 'high']):
        trajectories = results[grad]
        last_step = jnp.argmax(jnp.cumsum(trajectories.mask, axis=1) > 60_000, axis=1)
        x = trajectories.x[np.arange(n_trajectories), last_step]
        bx.hist(x[...,0], bins=bins, color=color, alpha=0.8, lw=0.5, edgecolor='black', density=True, label=label)

    bx.set(yticklabels=[])
    bx.legend(title='SNR', fontsize=7, title_fontsize=7, locl='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    bx.set_xlabel(r'Final displacement $\Delta x(t_{\text{end}})$ [\textmu m]')
    # Make x=50 show x=0
    bx.set_xticks(ticks=np.arange(xmin, xmax, 10), labels=np.arange(xmin-x0, xmax-x0, 10))
    plt.tight_layout()
    figname = 'chemotaxis_trajectories.pdf'
    print(f"Saving figure in {figures_dir / figname}")
    fig.savefig(figures_dir / figname, bbox_inches="tight", dpi=300)
    plt.close()
