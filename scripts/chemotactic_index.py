"""
This script generates the chemotactic index as a function of the signal-to-noise ratio.
"""
from functools import partial
import itertools
import pathlib

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit
from tqdm import tqdm

import psxc

matplotlib.use("Agg")

# Parameters and constants

SEED = 42
N_REPS = 1000
X_INIT = 50

GRADS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.75, 1.4, 1.8]
CVALS = [5, 25.0, 75.0, 125.0, 175.0]
ENVS = list(itertools.product(GRADS, CVALS))

CURRENT_DIR = pathlib.Path(__file__).parent

EXP_DATA = {
    "tweedy": {
        "path": CURRENT_DIR / "data/fig1c_scirep.dat",
        "label": "Tweedy et al. (2013)",
        "color": "black",
        "marker": "o",
        "markersize": 4,
    },
    "van_haastert": {
        "path": CURRENT_DIR / "data/fig_pnas.dat",
        "label": "van Haastert et al. (2007)",
        "color": "gray",
        "marker": "s",
        "markersize": 3,
    },
}


params = psxc.Parameters()

CONFIGS = {
    12: {
        "signal_fn": lambda _: jnp.ones(12),
        "label": r"Unsuppressed ($n{=}12$)",
        "color": "blue",
        "linestyle": "-",
    },
    2: {
        "signal_fn": lambda _: jnp.zeros(12).at[2].set(1.0).at[10].set(1.0),
        "label": r"Splitting ($n{=}2$)",
        "color": "red",
        "linestyle": "-",
    },
}

# Functions

def read_points_from_file(path):
    points = np.loadtxt(path, skiprows=1, delimiter=",")
    x, y = points[:, 0], points[:, 1]
    if points.shape[1] == 3:
        yerr = points[:, 2]
    else:
        yerr = np.zeros_like(y)
    return x, y, yerr


@partial(jax.jit, static_argnames=("signal_fn",))
def get_ci_and_snr(key, grad, value, signal_fn):

    profile = psxc.LinearProfile(grad, value, x=X_INIT)
    dt = 0.1
    run_fn = partial(
        psxc.run_chemotactic_trajectory,
        signal_fn=signal_fn,
        num_steps=25,
        warmup=3,
        t_max=30,
        dt=dt,
    )

    def collapse(x):
        return x.reshape(-1, *x.shape[2:])

    def _single_run(key, x0):
        key, key_cell = jax.random.split(key)
        cell_state = psxc.CellState.spawn(key_cell, x=x0, length=1.0)
        xs, mask = run_fn(params, key, cell_state, profile)
        xs, mask = jax.tree.map(collapse, (xs, mask))
        ci = psxc.weighted_chemotactic_index(xs)
        snr = psxc.signal_to_noise(xs, profile)

        # Compute the center of mass of the cell at each timestep
        pos = jnp.mean(xs, axis=(-2))
        msd = (pos[:, 0] - x0) ** 2
        time = jnp.cumsum(mask) * 0.1
        return ci, snr, mask, msd, time

    xs_init = jax.random.normal(key, (N_REPS,)) * 20 + X_INIT
    keys = jax.random.split(key, N_REPS)
    outs = jax.vmap(_single_run)(keys, xs_init)
    return jax.tree_util.tree_map(collapse, outs)


def to_numpy(tree):
    return jax.tree.map(np.asarray, tree)


if __name__ == '__main__':

    # Run the simulations
    plotting_data = {}
    for n_pods, config in CONFIGS.items():

        key = jax.random.key(SEED)

        metrics = ["ci", "snr", "msd", "time"]
        results = {}
        for n_start, (grad, value) in enumerate(tqdm(ENVS, ncols=80, desc=f"Running {n_pods}")):
            key_exp = jax.random.fold_in(key, n_start)
            ci, snr, mask, d, t = to_numpy(get_ci_and_snr(key_exp, grad, value, config["signal_fn"]))
            results[(grad, value)] = {"ci": ci[mask], "snr": snr[mask], "msd": d[mask], "time": t[mask]}

        sizes = {e: len(results[e]["snr"]) for e in results}
        total_size = sum(sizes.values())


        # Gather the results in flat arrays
        combined = {label: np.empty(total_size) for label in metrics}
        n_start = 0
        for env in ENVS:
            n = sizes[env]
            for metric in metrics:
                combined[metric][n_start : n_start + n] = results[env][metric]
            n_start += n

        # Only keep valid entries
        valid_entries = combined["snr"] < 0
        for metric in metrics:
            combined[metric] = combined[metric][valid_entries]

        # Bin the flat arrays
        x, y, yerr = psxc.put_on_bins(combined["snr"], combined["ci"], bins=np.linspace(-6.5, 0, 30))

        # Fit the data to a logistic function
        logistic_fn = lambda x, a, b, c: a / (1 + np.exp(-b * (x - c)))
        popt, *_ = curve_fit(logistic_fn, x, y, sigma=yerr, p0=[0.8, 1.0, -5])
        print("Fit parameters:", popt)

        x_fit = np.linspace(-6.5, 0, 1000, endpoint=False)
        y_fit = logistic_fn(x_fit, *popt)
        plotting_data[n_pods] = {"x": x, "y": y, "yerr": yerr, "x_fit": x_fit, "y_fit": y_fit, "popt": popt}

    # Load the experimental data
    for name, data in EXP_DATA.items():
        x, y, yerr = read_points_from_file(data["path"])
        plotting_data[name] = {"x": x, "y": y, "yerr": yerr}

    x_theory = np.linspace(-6.5, 0, 1000, endpoint=False)
    y_theory = 0.9 * psxc.perfect_absorber_limit(10**x_theory, a=50)

    # Plot the results

    FIGURES_DIR = CURRENT_DIR / pathlib.Path("figures")

    plt.style.use(["science"])

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0), dpi=200)
    ax.set_xlabel(r"Signal-to-Noise $\log_{10}$(SNR) [nM/\textmu m]")
    ax.set_ylabel(r"Chemotactic Index")

    ax.plot(x_theory, y_theory, ls="-", color="black", lw=1, label="Theorical limit")

    for name, data in plotting_data.items():
        if name in CONFIGS:
            x, y = data["x_fit"], data["y_fit"]
            color = CONFIGS[name]["color"]
            label = CONFIGS[name]["label"]
            linestyle = CONFIGS[name]["linestyle"]
            ax.plot(x, y, ls=linestyle, color=color, lw=2, label=label)
        elif name in EXP_DATA:
            x, y, yerr = data["x"], data["y"], data["yerr"]
            color = EXP_DATA[name]["color"]
            label = EXP_DATA[name]["label"]
            marker = EXP_DATA[name]["marker"]
            ms = EXP_DATA[name]["markersize"]
            plot_props = {"marker": marker, "color": color, "ls": "", "capsize": 2, "mec": color, "ms": ms}
            ax.errorbar(x, y, yerr=yerr, **plot_props, label=label)

    ax.set_xlim(-6.5, 0)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(-6, 1, 1), [fr"$10^{{{i}}}$" for i in np.arange(-6, 1, 1)])

    # Split the handles and labels of the legend in simulation and experimental data
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    is_data_handle = [isinstance(h, matplotlib.container.ErrorbarContainer) for h in ax_handles]
    lines_handles = [h for h, is_data in zip(ax_handles, is_data_handle) if not is_data]
    lines_labels = [l for l, is_data in zip(ax_labels, is_data_handle) if not is_data]
    legend = ax.legend(lines_handles, lines_labels, fontsize=8, loc="upper left")
    ax.add_artist(legend)
    data_handles = [h[0] for h, is_data in zip(ax_handles, is_data_handle) if is_data]
    data_labels = [l for l, is_data in zip(ax_labels, is_data_handle) if is_data]
    ax.legend(data_handles, data_labels, fontsize=8, loc="lower right")
    plt.tight_layout()

    # Save the figure
    figname = "chemotactic_index_signal_to_noise.pdf"
    fig.savefig(FIGURES_DIR / figname, bbox_inches="tight")

    # Save results of simulations
    save_data = {}
    for n_pods in CONFIGS:
        label = CONFIGS[n_pods]["label"]
        color = CONFIGS[n_pods]["color"]
        linestyle = CONFIGS[n_pods]["linestyle"]
        popt = plotting_data[n_pods]["popt"]
        save_data[n_pods] = {"label": label, "color": color, "popt": list(popt)}

    import json
    with open(CURRENT_DIR / "data" / "ci_results.json", "w") as f:
        f.write(json.dumps(save_data, indent=4))
