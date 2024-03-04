"""
This script performs a series of simulations to study the speed-accuracy trade-off in the context of the Weber's Law.
The results are saved in the data folder.

Usage:
    python weber_law.py [-s] [-nc]

Options:
    -s, --skip-calculations: Skip calculations and use the data in the data folder.
    -nc, --noclean: Do not clean the data folder.

Author: Albert Alonso
Date: 2024.03.24
"""
import argparse
import functools
import itertools
import pathlib

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import scienceplots  # pyright: ignore
from labellines import labelLine, labelLines

import psxc

matplotlib.use("Agg")

parser = argparse.ArgumentParser(description="Speed vs Accuracy Decision")
parser.add_argument("--skip-calculations", "-s", action="store_true", help="Skip calculations")
parser.add_argument("--noclean", "-nc", action="store_true", help="Do not clean the data")
args = parser.parse_args()

# PARAMETERS
SEED = 0
x0 = 50
dt = 0.1
grads = np.linspace(0.01, 1.8, 100)
values = np.linspace(5, 175, 100)
experiments = list(itertools.product(grads, values))

exchange_rates = [0.4, 0.5, 0.6, 0.7]
less_grads = np.linspace(0.01, 1.8, 100)
less_values = np.linspace(15, 175, 10)
less_experiments = list(itertools.product(less_grads, less_values))

current_file = pathlib.Path(__file__).stem
current_folder = pathlib.Path(__file__).parent

datadir = current_folder / "data" / current_file
datadir.mkdir(parents=True, exist_ok=True)

if not args.skip_calculations:

    @functools.partial(jax.jit, static_argnames=("nreps", "dt"))
    def evaluate(key, grad, val, params, nreps=1000, dt=0.1):

        profile = psxc.LinearProfile(grad, val, x=x0)

        def calc_decision(key):
            key_init, key_sim = jax.random.split(key, num=2)
            cell_state = psxc.CellState.spawn(key_init, x=x0)
            (xs, mask), xf, T = psxc.sensing_event(params, key_sim, cell_state, profile, t_max=20.0, dt=dt)
            Td, Tg = psxc.decision_time(xs, mask, dt=dt)
            return {"acc": xf[0] > cell_state.xf[0], "time": Td}

        keys = jax.random.split(key, nreps)
        return jax.tree_util.tree_map(jnp.mean, jax.vmap(calc_decision)(keys))

    key = jax.random.key(SEED)

    print("Performing first experiment...")
    results1 = {}
    params = psxc.Parameters(noise_strength=1.5e-3)
    mapping = {(g, v): (i, j) for i, g in enumerate(grads) for j, v in enumerate(values)}
    results = np.zeros((len(grads), len(values), 2))
    for i, (grad, val) in enumerate(tqdm(experiments, ncols=80)):
        key_exp = jax.random.fold_in(key, i)
        outs = evaluate(key_exp, grad, val, params, 100_000)
        results[mapping[(grad, val)]] = outs["acc"], outs["time"]
    results1 = {"acc": results[..., 0], "time": results[..., 1], "grads": grads, "values": values}

    # Save the results
    print("Saving results...", end="\r")
    if not args.noclean:
        {file.unlink() for file in datadir.glob(f"{current_file}_*.npz")}
    exp_filename = f"{current_file}_heatmap.npz"
    np.savez(datadir / exp_filename, **results1)
    print("Result saved at:", datadir)

    # Second experiment: How do the minimum value changes for different exchange rate
    print("Performing second experiment...")
    results2 = {}
    mapping = {(g, v): (i, j) for i, g in enumerate(less_grads) for j, v in enumerate(less_values)}
    for e in tqdm(exchange_rates, ncols=80):
        params = psxc.Parameters(exchange_rate=e)
        results = np.zeros((len(less_grads), len(less_values), 2))
        for i, (grad, val) in enumerate(tqdm(less_experiments, ncols=80, leave=False)):
            key_exp = jax.random.fold_in(key, i)
            outs = jax.tree_util.tree_map(np.asarray, evaluate(key_exp, grad, val, params, 100_000))
            results[mapping[(grad, val)]] = outs["acc"], outs["time"]

        acc, times = results[..., 0], results[..., 1]
        results2[e] = {"acc": acc, "time": times, "grads": less_grads, "values": less_values}

    print("Saving results...", end="\r")
    for e, data in results2.items():
        exp_filename = f"{current_file}_exchange_{e:.5g}.npz"
        np.savez(datadir / exp_filename, **data)
    print("Result saved at:", datadir)

else:
    print(f"Loading results from {datadir}")

    filename = next(datadir.glob(f"{current_file}_heatmap*.npz"))
    results1 = np.load(filename)

    results2 = {}
    files = list(datadir.glob(f"{current_file}_exchange*.npz"))
    for filename in tqdm(files, ncols=80):
        exchange_rate = float(filename.stem.split("_")[-1])
        results2[exchange_rate] = np.load(filename)

# ------------------------------
# Plot the results
# ------------------------------
figures_dir = current_folder / pathlib.Path("figures")
figures_dir.mkdir(exist_ok=True, parents=True)

plt.style.use(["science"])

data_min = {}
data_trans = {}
for e in exchange_rates:
    acc = results2[e]["acc"]
    time = results2[e]["time"]
    less_grads = results2[e]["grads"]
    less_values = results2[e]["values"]

    idx = np.argmax(acc >= 0.95, axis=0)
    minimum_gradient = less_grads[idx]
    time_decision = time[idx, np.arange(len(less_values))]
    data_min[e] = {
        "x": less_values[idx > 0],
        "y": minimum_gradient[idx > 0],
        "t": time_decision[idx > 0],
    }

    # Get one vertical line
    y = acc[:, 5]
    x = less_grads
    data_trans[e] = {"x": x, "y": y}

# Get the data for the exchange rate dependency
accuracy = results1["acc"]
times = results1["time"]
values = results1["values"]
grads = results1["grads"]

fig, ax = plt.subplots(1, 1, figsize=(5, 4.0), dpi=200)
ax.set(xlabel=r"Concentration $c_0$ [nM]", ylabel=r"Gradient $\nabla c$ [nM/\textmu m]")
cax = ax.pcolormesh(values, grads, accuracy, cmap="inferno", vmin=0.5, lw=0, rasterized=True)
cb = fig.colorbar(cax, orientation="vertical", label="Decision Correctness")
cb.ax.tick_params(labelsize=8)
cs = ax.contour(values, grads, results1["acc"], levels=[0.95], colors="blue", linewidths=1, linestyles="-")
idx = np.argmax(results1["acc"] >= 0.95, axis=0)
y = grads[idx][values>30]
x = values[values>30]
# Fit to a line
fit_fn = lambda x, a, b: a * x + b
popt, _ = curve_fit(fit_fn, x, y)
print(popt)
x = np.linspace(values[0], values[-1], 100)
ax.plot(x, fit_fn(x, *popt), ls="-", lw=1.5, zorder=1, color='black')
ax.axvline(30, color="black", linestyle="-", lw=1.5)

# Add an inset
bx = ax.inset_axes([0.061, 0.55, 0.5, 0.4])

#ax.clabel(CS, inline=True, fontsize=9, fmt="%.2g")

#fig, axes = plt.subplots(2, 2, figsize=(5, 4.5), dpi=200, sharey=False,
#                         sharex=False, gridspec_kw={"height_ratios": [1, 1]})
#
#
#ax1 = axes[0, 0]
#ax1.set(xlabel=r"Concentration $c_0$ [nM]", ylabel=r"Gradient $\nabla c$ [nM/\textmu m]")
#ax1.set_title("(a) Correct Decision Rate", fontsize=9, loc="left")
#cax = ax1.pcolormesh(values, grads, accuracy, cmap="inferno", vmin=0.5, lw=0, rasterized=True)
#cb = fig.colorbar(cax, orientation="vertical")
#cb.ax.tick_params(labelsize=8)
#CS = ax1.contour(values, grads, results1["acc"], levels=[0.95, 0.99], colors="black", linewidths=1.8)
#ax1.clabel(CS, inline=True, fontsize=9, fmt="%.2g")
#
#
#ax2 = axes[1, 0]
#ax2.set(xlabel=r"Concentration $c_0$ [nM]", ylabel=r"Gradient $\nabla c$ [nM/\textmu m]")
#ax2.set_title("(c) Decision Time $T_D$ [s]", fontsize=9, loc="left")
#cax = ax2.pcolormesh(values, grads, results1["time"], cmap="Oranges", lw=0, rasterized=True)
#cb = fig.colorbar(cax, orientation="vertical")
#CS = ax2.contour(values, grads, times, levels=[7, 8, 9, 10, 11, 12], colors="black", linewidths=1.0)
#ax2.clabel(CS, inline=True, fontsize=8, fmt="%.2g")
#
#
#bx = axes[0, 1]
#gc = bx.get_gridspec()
#bx = fig.add_subplot(gc[1, :], sharey=ax1)
#bx.text(0.05, 0.9, "(c)", fontsize=9, transform=bx.transAxes, va="top", ha="left")
#bx.set(xlabel=r"Concentration $c_0$ [nM]")
#bx.set_title("(b) Minimum Gradient Required", fontsize=9, loc="left")
colors = {0.4: 'black', 0.5: 'blue', 0.6: 'red', 0.7: 'green'}
for m, e in zip("vosD^", exchange_rates):
    num_points = len(data_min[e]["x"])
    x, y, t = data_min[e]["x"], data_min[e]["y"], data_min[e]["t"]
    #cax = bx.scatter(x, y, c='w', marker=m, s=10, edgecolors="black", label=rf"${e}$")
    sqrt_fn = lambda x, a, b: np.sqrt(x) * a + b
    popt, _ = curve_fit(sqrt_fn, data_min[e]["x"], data_min[e]["y"])
    x = np.linspace(values[0], values[-1], 100)
    bx.plot(x, sqrt_fn(x, *popt), ls="-", lw=2.0, zorder=0, label=rf"${e}$", color=colors[e])
#labelLines(bx.get_lines(), zorder=2.5, fontsize=9, shrink_factor=0.7)
bx.legend(ncols=2, labelspacing=0.2, columnspacing=1.0, fontsize=8, title=r'$\varepsilon$')
bx.axvline(30, color="black", linestyle="-", lw=1.5)
bx.set_ylim(0, 1.8)
bx.tick_params(labelbottom=False, labelleft=False)
#
## Plot the vertical line at a given concentration
#bx2 = axes[1, 1]
#bx2.set(xlabel=r"Gradient $\nabla c$ [nM\textmu m]")
#bx2.set_title("(d) Correct Probability", fontsize=9, loc="left")
#for e in exchange_rates:
#    x, y = data_trans[e]["x"], data_trans[e]["y"]
#
#    bx2.plot(x, y, lw=2, zorder=0, label=rf"${e}$", color=colors[e])
##labelLines(bx2.get_lines(), zorder=2, fontsize=9)
#bx2.legend()
#
#
plt.tight_layout()
figname = f"weber_law_combined.pdf"
fig.savefig(figures_dir / figname, bbox_inches="tight")
print("Figure saved at:", figures_dir / figname)
