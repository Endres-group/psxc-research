from functools import partial
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from scipy.spatial import ConvexHull
from matplotlib_scalebar.scalebar import ScaleBar
from concurrent.futures import ProcessPoolExecutor, as_completed
import ffmpeg
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

import psxc


def collapse(x):
    return x.reshape(-1, *x.shape[2:])


dt = 0.1

def trajectory(key, params, profile, x0, signal_fn, y0=0.0):
    splitting_event = partial(psxc.run_chemotactic_trajectory, num_steps=5, warmup=0, t_max=50, dt=dt)

    key, key_cell = jax.random.split(key)
    cell_state = psxc.CellState.spawn(key_cell, x=x0, length=1.0, y=y0)
    xs, mask = splitting_event(params, key, cell_state, profile, signal_fn)
    xs, mask = jax.tree.map(collapse, (xs, mask))
    return xs, mask

key = jax.random.key(32)
x0 = 50.0

length = 1.0
params = psxc.Parameters(exchange_rate=0.4)
signal = lambda _: jnp.ones(12)
results = {}

#params = psxc.Parameters(activation_rate=2, noise_strength=1.5e-3, exchange_rate=0.3)
#signal = lambda _: jnp.zeros(12).at[2].set(1.0).at[10].set(1.0)
profile = psxc.LinearProfile(0.01, 100, x=x0)
xs, mask = trajectory(key, params, profile, x0, signal, y0=0.6)
xs = np.asarray(xs[mask])[::5]
results['compass'] = np.copy(xs)

params = psxc.Parameters()
signal = lambda _: jnp.zeros(12).at[2].set(1).at[10].set(1)
xs, mask = trajectory(key, params, profile, x0, signal, y0=-0.6)
xs = np.asarray(xs[mask])[::5]
#results['splitting'] = np.copy(xs)

dt = dt * 5


# Simulations are done

frames_path = pathlib.Path(__file__).parent / 'data' / 'frames'
frames_path.mkdir(exist_ok=True, parents=True)
# remove all files in the folder
[fp.unlink() for fp in frames_path.glob('*.png')]

data = {}
for key in results:
    xs = results[key]
    *pp, pc, pu = np.asarray(xs).transpose(1, 0, 2)
    pp.append(pu)
    pp = np.array(pp).swapaxes(0, 1)
    pc = np.array(pc)
    pu = np.array(pu)

    PAD = 2
    # Find the limits
    xmin = np.min(pc[:,0]) - PAD
    xmax = np.max(pc[:,0]) + PAD
    ymin = np.min(pc[:,1]) - PAD
    ymax = np.max(pc[:,1]) + PAD
    data[key] = (pp, pc, (xmin, xmax, ymin, ymax))

# Find the total limits
xmin = min([d[2][0] for d in data.values()])
xmax = max([d[2][1] for d in data.values()])
ymin = min([d[2][2] for d in data.values()])
ymax = max([d[2][3] for d in data.values()])
bounds = (xmin, xmax, ymin, ymax)
#bounds = (49, 63, -4.0, 4.0)
cbounds = (90.0, 110)

print(bounds)
#bounds = (45, 55, -3, 3)
xmin, xmax, ymin, ymax = bounds

xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
c = jax.vmap(jax.vmap(profile.concentration))(np.dstack([xx, yy]))
c = np.asarray(c)
print(c.min(), c.max(), c.mean())

def plot_environment(t):

    def find_body_outline(points):

        def make_body(x, radius=1.0):
            theta = np.linspace(0, 2 * np.pi, 100)
            return np.array([x[0] + radius * np.cos(theta), x[1] + radius * np.sin(theta)]).T

        body = np.concatenate([make_body(p, radius=0.15) for p in points])
        hull = ConvexHull(body)
        body = body[hull.vertices]
        return body

    fig, ax = plt.subplots(1, 1, dpi=200)
    xmin, xmax, ymin, ymax = bounds

    colormesh = ax.pcolormesh(xx, yy, c, cmap='Blues', rasterized=True, vmin=cbounds[0], vmax=cbounds[1])
    # Add a small horizontal colorbar inside the plot using a subaxes at the lower left part
    cax = fig.add_axes([0.70, 0.15, 0.02, 0.2])
    fig.colorbar(colormesh, cax=cax, orientation='vertical')



    for key in results:
        x = results[key][t]
        pp, pc = data[key][:2]
        ax.plot(pc[:t][0], pc[:t][1], '-', color='red', ms=3, zorder=5, alpha=0.5)
        pp, pc = pp[t], pc[t]
        body = find_body_outline(x)
        p = patches.Polygon(body, closed=True, fc='white', alpha=0.7, animated=True, ec='k', fill=True)
        ax.add_patch(p)
        pods = [ax.plot([p[0], pc[0]], [p[1], pc[1]], '-', lw=1, color='black', solid_capstyle='round', ms=2, zorder=6)[0] for p in pp]
        pods[-1].set_color('gray')
        pods[-1].set_zorder(5)

    hours = int(np.floor(t * dt / 3600))
    minutes = int(np.floor((t * dt - hours * 3600) / 60))
    seconds = int(t * dt - hours * 3600 - minutes * 60)
    #text = ax.text(0.85, 0.95, f'[{hours:02d}:{minutes:02d}:{seconds:02d}]', ha='center', va='center', transform=ax.transAxes, color='w', animated=True)
    #scalebar = ScaleBar(10, "um", box_color='None', color='w', location='lower right', width_fraction=0.04, frameon=True)
    #ax.add_artist(scalebar)
    text = ax.text(0.1, 0.96, 'High SNR (Pseudopod competition)', ha='left', va='top', transform=ax.transAxes, color='k', animated=True, fontsize=10)
    ax.set(xticks=[], yticks=[])

#    ax.set_xlim(xmin, xmax)
#    ax.set_ylim(ymin, ymax)
    ax.set_xlim(pp.mean(0)[0] - 2, pp.mean(0)[0]+ 2)
    ax.set_ylim(pp.mean(0)[1] - 2, pp.mean(0)[1] + 2)

    ax.set_aspect('equal', 'box')
    fig.savefig(frames_path / f'{t:06d}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

with ProcessPoolExecutor(64) as executor:
    tasks = [executor.submit(plot_environment, i) for i in range(len(xs))]
    total = len(xs)

    for result in tqdm(as_completed(tasks), total=total, ncols=81):
        pass

figdir = pathlib.Path(__file__).parent / 'figures'
figdir.mkdir(exist_ok=True, parents=True)
ffmpeg.input(str(frames_path / "*.png"), pattern_type='glob', framerate=10).output(str(figdir / "animation.mp4"), pix_fmt='yuv420p', vf='scale=trunc(iw/2)*2:trunc(ih/2)*2').run(quiet=True, overwrite_output=True)
#ffmpeg.input(str(frames_path / "*.png"), pattern_type='glob', framerate=2).output(str(figdir / "animation.mp4")).run(quiet=False, overwrite_output=True)

