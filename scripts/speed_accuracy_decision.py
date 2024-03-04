"""
This script calculates the speed vs accuracy decision for different number of pseudopods.
The results are saved in the data folder.
Usage:
    python speed_accuracy_decision.py [-s] [-d datafile] [-nc]

Options:
    -s, --skip-calculations: Skip calculations and use the data in the data folder.
    -d, --datafile: Datafile to save the results.
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

import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm, trange
import scienceplots  # pyright: ignore
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import matplotlib

matplotlib.use("Agg")  # This is due to running it headless on a remote computer
import psxc

parser = argparse.ArgumentParser(description="Speed vs Accuracy Decision")
parser.add_argument("--skip-calculations", "-s", action="store_true", help="Skip calculations")
parser.add_argument("--noclean", "-nc", action="store_true", help="Do not clean the data")
args = parser.parse_args()

# Parameters
SEED = 0
x0 = 50.0
dt = 0.1
grads = np.linspace(0.01, 2, 50)
# grads = np.logspace(np.log10(0.01), np.log10(2), 10)
values = [75.0]
experiments = list(itertools.product(grads, values))
setups = {
    2: [3, 9],
    3: [0, 4, 8],
    4: [0, 3, 6, 9],
    5: [0, 2, 5, 7, 10],
    6: [0, 2, 4, 6, 8, 10],
    7: [0, 2, 3, 5, 7, 9, 10],
    8: [0, 2, 3, 5, 6, 7, 9, 10],
    9: [0, 2, 3, 4, 5, 7, 8, 9, 10],
    10: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    11: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
    12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

CURRENT_FILE = pathlib.Path(__file__).stem
CURRENT_FOLDER = pathlib.Path(__file__).parent

params = psxc.Parameters()

if not args.skip_calculations:
    print("Performing calculations... Use -s to skip")

    @functools.partial(jax.jit, static_argnames=("num_reps", "dt"))
    def evaluate(key, grad, value, params, signal_strength, num_reps=1000, dt=dt):
        """Function that runs num_reps independent splitting events, and returns the CI
        and duration of each."""

        profile = psxc.LinearProfile(grad=grad, value=value, x=x0)

        def cosine_similarity(a: jax.Array, b: jax.Array) -> jax.Array:
            (x, y) = a - b
            return jnp.cos(jnp.arctan2(y, x))

        def calc_accuracy(key):
            key_theta, key_sim = jax.random.split(key)
#            key_theta = jax.random.key(42); print('Warning: Key theta is fixed for debugging')
            cell_state = psxc.CellState.spawn(key_theta, x=x0)
            (xs, mask), xf, _ = psxc.sensing_event(params, key_sim, cell_state, profile, signal_strength, t_max=100.0, dt=dt)
            td, _ = psxc.decision_time(xs, mask, alpha=0.01, dt=dt)
            ci = cosine_similarity(xf, cell_state.xf)
            return dict(ci=ci, duration=td, valid=mask)

        keys = jax.random.split(key, num=num_reps)
        return jax.vmap(calc_accuracy)(keys)

    # Iterate for different ammount of pseudopods
    key = jax.random.key(SEED)
    results = {}
    for n in (bar := trange(2, 13, ncols=80)):
        bar.set_description(f"Number of pseudopods: {n}")
        if n not in setups:
            continue
        ind = setups[n]
        signal_strength = jnp.zeros(12).at[np.array(ind)].set(1.0)
        data = {}
        for i, (g, v) in enumerate((expbar := tqdm(experiments, ncols=80, leave=False))):
            if n != 12 and g != 2.0:
                continue
            expbar.set_description(f"Gradient: {g:.5g}, Value: {v:.5g}")
            ikey = jax.random.fold_in(key, i)
            outs = evaluate(ikey, g, v, params, signal_strength, num_reps=10_000)
            data[(g, v)] = jax.tree.map(np.asarray, outs)
        results[len(ind)] = data

    # Save the results in datafile
    print("Saving results...", end="\r")
    datadir = CURRENT_FOLDER / (pathlib.Path("data/") / CURRENT_FILE)
    datadir.mkdir(exist_ok=True, parents=True)
    if not args.noclean:
        {file.unlink() for file in datadir.glob(f"{CURRENT_FILE}_*.npz")}
    for n, exp in results.items():
        for (g, v), dat in exp.items():
            exp_filename = datadir / f"{CURRENT_FILE}_{n}_{g:.5g}_{v:.5g}.npz"
            np.savez(exp_filename, **dat)
    print("Results saved at:", datadir)

else:
    # Load the files into results dict
    datadir = CURRENT_FOLDER / (pathlib.Path("data/") / CURRENT_FILE)
    files = list(datadir.glob(f"{CURRENT_FILE}_*.npz"))
    num_pods = sorted(list(set([int(file.stem.split("_")[3]) for file in files])))
    grads = sorted(list(set([float(file.stem.split("_")[4]) for file in files])))
    values = sorted(list(set([float(file.stem.split("_")[5]) for file in files])))
    experiments = list(itertools.product(grads, values))
    results = {}
    for n in (bar := tqdm(num_pods, ncols=80)):
        bar.set_description(f"Number of pseudopods: {n}")
        data = {}
        for (g, v) in itertools.product(grads, values):
            filename = datadir / f"{CURRENT_FILE}_{n}_{g:.5g}_{v:.5g}.npz"
            data[(g, v)] = np.load(filename)
        results[n] = data

# ------------------------------
# Plot the results
# ------------------------------
print("Plotting results...")

FIGDIR = CURRENT_FOLDER / pathlib.Path("figures")
FIGDIR.mkdir(exist_ok=True, parents=True)

# Plot the accuracy of the n=12 pseudopds
with plt.style.context(["science"]):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.0, 5.0), dpi=300, sharey=True, height_ratios=[1.2, 1])

    points = {}
    cis = []
    gs = []
    durations = []
    metrics = []
    for (g, v) in experiments:
        data = results[12][(g, v)]
        ci = data["ci"]
        cis.extend(ci)
        gs.extend([g] * len(ci))
        durations.extend(data["duration"])
        metrics.append((ci.mean(), ci.std() / np.sqrt(len(ci)), len(ci)))
        print(f'Duration: {data["duration"].mean()}')

    means, stds, counts = np.array(metrics).T
    T = np.mean(durations)

    #*_, image = ax.hist2d(gs, cis, bins=(len(grads), 50), cmap="magma_r", norm=matplotlib.colors.LogNorm(), lw=0)
    #ax.set(facecolor="black")
    #err = ax.errorbar(g, mean, yerr=std, marker=marker, ls='none', color=color, capsize=2, elinewidth=1, ms=5, mfc='w')
    # Insert the colorbar inside the plot (lower right)
    #axins = ax.inset_axes([0.5, 0.1, 0.4, 0.05])
    #cb = plt.colorbar(image, ax=ax, cax=axins, orientation="horizontal")
    # Normalize the counts by the number of values on each gradient (
    #cb.set_ticks([1e-3, ], labels=[0, 1])

    #ax.plot(grads, means, color="green", lw=1.0, ls="none", marker='.', ms=4, mew=0.5, label='Simulations')
    ax.errorbar(grads[::2], means[::2], yerr=stds[::2], color="blue", ls='none', marker='o', ms=4, capsize=2, elinewidth=1, zorder=2, label='Simulations')

    f = lambda x, a: 0.9 * psxc.theorical_limit_similarity(x, value=75.0, a=a, T=T)
    popt, _ = curve_fit(f, grads, means, p0=[1.0])
    print("Resulting parameters: ", "a=", popt, "T=", T)
    for A, linestyle, label in zip([1.0, 0.9], ["-", "--"], ['Analytical', 'Adapted']):
        y = A * psxc.theorical_limit_similarity(grads, value=75.0, a=popt[0], T=T)
        ax.plot(grads, y, color="black", linestyle=linestyle, lw=1.2, zorder=3, label=label)


    inset = ax.inset_axes([0.5, 0.30, 0.4, 0.4])
    hist1, bin_edges_1 = np.histogram(results[12][(min(grads), 75.0)]["ci"], bins=np.linspace(-1, 1, 100))
    hist2, bin_edges_2 = np.histogram(results[12][(max(grads), 75.0)]["ci"], bins=np.linspace(-1, 1, 100))
    for gradient, color in zip([min(grads), max(grads)], ["green", "blue"]):
        x = np.linspace(-1, 1, 1000)
        f = gaussian_kde(results[12][(gradient, 75.0)]["ci"]).pdf(x)
        #inset.plot(x, f/max(f), color=color, lw=1.0)
        #inset.fill_between(x, f/max(f), color=color, alpha=0.2, label=f"{gradient:.0g}")
        inset.hist(results[12][(gradient, 75.0)]["ci"], bins=np.linspace(-1, 1, 10), rwidth=0.8, color=color, lw=1.0, label=f"{gradient:.0g}", alpha=0.4)
    inset.tick_params(labelleft=False)
    inset.legend(ncols=1, fontsize=7, title=r"$\nabla c$ [nM/\textmu m]", title_fontsize=7)
    ax.legend(ncols=2)

    ax.set_xlabel(r"Gradient $\nabla c$ [nM/\textmu m]")
    ax.set_ylabel(r"Alignment $\langle GA\rangle$")

    # Pseudopod
    n_pseudopods = sorted(list(setups.keys()))
    g, v = 2.0, 75.0
    metrics = []
    for n in n_pseudopods:
        data = results[n][(g, v)]
        time = data["duration"].mean()
        acc = data["ci"].mean()
        acc_ste = data["ci"].std() / np.sqrt(len(data["ci"]))
        time_ste = data["duration"].std() / np.sqrt(len(data["duration"]))
        metrics.append((n, acc, acc_ste, time, time_ste))

    metrics = np.array(metrics)
    n_pods, acc, acc_ste, time, time_ste = metrics.T
    #bx.plot(n_pods, acc, marker='o', ls='-', lw=1.0, color='black', label='Accuracy')
    bx.errorbar(n_pods, acc, yerr=acc_ste, color='black', ls='-', marker='o', ms=4, capsize=2, elinewidth=1, label="Alignment")
    bx.set_xlabel(r"Number of pseudopods $n$")
    bx.set_ylabel(r"Alignment $\langle GA\rangle$")

    # Add another label to the right
    bx2 = bx.twinx()
    bx2.errorbar(n_pods, time, yerr=time_ste, color='red', ls='-', marker='D', ms=4, capsize=2, elinewidth=1, label='Decision Time')
    bx2.set_ylabel(r"$\langle T_D \rangle$ [s]", color='red')
    bx2.tick_params(axis='y', labelcolor='red')
    #bx2.set_ylim(0, 2)

    inset = bx.inset_axes([0.5, 0.2, 0.4, 0.4])
    f = lambda x, a, b, c: a * x**(-b) + c
    eff_err = 1/time * (acc_ste + acc/time * time_ste)
    popt, cov = curve_fit(f, n_pods, acc/time, p0=[3.0, 0.5, 0.0])#, sigma=eff_err)
    # Get the error of b
    perr = np.sqrt(np.diag(cov))
    print(popt)
    print(perr)
    inset.errorbar(n_pods, acc/time, yerr=eff_err, marker='.', ls='none', color='black', capsize=2, elinewidth=1, ms=4)
    x = np.linspace(2, 12, 100)
    y = f(x, *popt)
    inset.plot(x, y, color='black', linestyle='--', lw=0.5)
    inset.text(0.5, 0.5, fr"$ \nu = {{{popt[1]:.1f}}}\pm{{{perr[1]:.1f}}}$", transform=inset.transAxes, ha='center', va='center', fontsize=8)
    inset.set_xlabel(r"$n$")
    inset.set_ylabel(r"$\langle GA \rangle / \langle T_D \rangle$")

    #inset.plot(bin_edges_1[:-1], hist1/max(hist1), color='green', lw=1.0)
    #inset.plot(bin_edges_2[:-1], hist2/max(hist2), color='red', lw=1.0)
    #inset.hist(results[12][(min(grads), 75.0)]["ci"], bins=np.linspace(-1, 1, 100), histtype='step', color='black')
    #inset.hist(results[12][(max(grads), 75.0)]["ci"], bins=np.linspace(-1, 1, 100), histtype='step', color='blue')

#        duration = data["duration"]
#        breaks = np.arange(2, 12+5, 5)
#        handles_and_labels = {'handles': [], 'labels': []}
#        for (minval, maxval, marker, color) in zip(breaks[:-1], breaks[1:], 'osv^d', 'bgkrcm'):
#            mask = (duration > minval) & (duration <= maxval)
#            if np.sum(mask) == 0:
#                continue
#            mean = np.mean(ci[mask])
#            std = np.std(ci[mask]) / np.sqrt(np.sum(mask))
#            _g = g
#            handles_and_labels['handles'].append(err)
#            handles_and_labels['labels'].append(f"{(maxval+minval)/2} ({minval}-{maxval})")
#            #ax.plot(x, y, color=color, linestyle="-", lw=1.0, label=f"{minval}-{maxval} s")
#    handles_without_line = [h[0] for h in handles_and_labels['handles']]
#    handles_and_labels['handles'] = handles_without_line
#    ax.legend(handles_and_labels['handles'], handles_and_labels['labels'], title="$T_D$ [s]", fontsize=8, title_fontsize=8)
#
#    x = np.linspace(0, 2, 1000)
#    for (minval, maxval, marker, color) in zip(breaks[:-1], breaks[1:], 'osv^d', 'bgkrcm'):
#        T = (minval + maxval) / 2
#        y = psxc.theorical_limit_similarity(x, value=75.0, a=2.0, T=T)
#        ax.plot(x, y, color=color, linestyle="-", lw=1.0, label=f"{T} ({minval}-{maxval})")
#        ax.plot(x, 0.9*y, color=color, linestyle="--", lw=1.0)



#    ax.set_ylim(-0.1, 1.1)

    # The second plot shows the error rate with the time.
    # We define the error as CI being positive or negative. Negative means error.
#    bx.set_xlabel('Decision Time $T_D$ [s]')
#    bx.set_ylabel('Error rate')
#    errors_results = {'time': [], 'error': []}
#    for (g, v) in experiments:
#        if g != grads[3]:
#            continue
#        data = results[12][(g, v)]
#        ci = data["ci"]
#        duration = data["duration"]
#        valid = (duration < 14) & (duration > 1)
#        duration = duration[valid]
#        ci = ci[valid]
#        errors_results['time'].extend(duration)
#        errors_results['error'].extend(1 * (ci < 0))
#    print(min(errors_results['time']), max(errors_results['time']))
#    x, y, yerr = psxc.put_on_bins(errors_results['time'], errors_results['error'], bins=np.linspace(4, 13, 10))
#    bx.errorbar(x, y, yerr=yerr, fmt='o', color='k', mfc='w', capsize=2, elinewidth=1, ms=5)

    labels  = ["a", "b"]
    for label, cx in zip(labels, [ax, bx]):
        cx.text(0.02, 0.97, fr'\textbf{{({label})}}', transform=cx.transAxes, weight='bold', va='top', ha='left', fontsize=11)
    plt.tight_layout()
    figure_name = "accuracy.pdf"
    fig.savefig(FIGDIR / figure_name, bbox_inches="tight")
    print("Figure saved at:", FIGDIR / figure_name)
#
## Plot about which pseudopods are active in each setup
#with plt.style.context(["science"]):
#    fig, axes = plt.subplots(2, 6, figsize=(5.0, 2.0), dpi=200, sharex="col")
#
#    for n, ax in enumerate(axes.flatten(), start=1):
#        ax.axis("off")
#        ax.set_ylim(-1.4, 1.4)
#        ax.set_xlim(-1.4, 1.4)
#        if n not in setups:
#            ax.annotate(
#                "Cell\nOrientation",
#                xy=(0.5, 0.8),
#                xytext=(0.5, -0.5),
#                ha="center",
#                va="center",
#                fontsize=8,
#                arrowprops=dict(arrowstyle="->"),
#            )
#            continue
#
#        theta = np.linspace(0, 2 * np.pi, 13)[:-1] + np.pi / 2
#        x = np.cos(theta)
#        y = np.sin(theta)
#        ax.plot(x, y, "o", color="white", mec="black", markersize=4)
#        x_active = x[setups[n]]
#        y_active = y[setups[n]]
#        ax.plot(x_active, y_active, "o", color="black", markersize=4)
#        for i in range(len(x_active)):
#            ax.plot([0, x_active[i]], [0, y_active[i]], color="black", lw=0.8)
#
#        ax.set_title(f"$n{{=}}{n}$", fontsize=8)
#        ax.set_aspect("equal")
#
#    plt.tight_layout()
#    figure_name = "manual_pseudopods_orientation.pdf"
#    plt.savefig(figures_dir / figure_name, bbox_inches="tight")
#    print("Figure saved at:", figures_dir / figure_name)
#    plt.close(fig)
#
#
#with plt.style.context(["science"]):
#
#    fig, ((ax1, bx1), (ax2, bx2)) = plt.subplots(2, 2, figsize=(5.0, 3.0), dpi=200, sharex="col")
#
#    ax1.set_title(r"(a) Gradient Alignment $GA$", fontsize=9, loc="left")
#
#    for marker, num_pods in zip("osDv^", [2, 3, 6, 12]):
#        data = results[num_pods]
#        y = [np.mean(data[(g, 75.0)]["ci"]) for g in grads]
#        ax1.plot(grads, y, ls="--", lw=0.5, zorder=3, color="black")
#        ax1.errorbar(
#            grads,
#            y,
#            marker=marker,
#            ls="",
#            lw=0.7,
#            markersize=4,
#            capsize=2,
#            elinewidth=1,
#            color="black",
#            mfc="w",
#            mew=0.8,
#            zorder=3,
#            label=rf"${num_pods}$",
#        )
#
#    _x = np.linspace(0.01, 2, 100)
#    theory = psxc.theorical_limit_similarity(_x, 75, a=80, T=2)
#    ax1.plot(_x, theory, color="black", linestyle="-", lw=0.8, zorder=0)
#
#    ax1.set_ylim(-0.1, 1.1)
#    # remove line in handles with markers and lines
#    #    handles, labels = ax1.get_legend_handles_labels()
#    #    handles = [h.get_marker() for h in handles]
#    ax1.legend(fontsize=8, ncols=4, columnspacing=0.2, handletextpad=0.2, title="$n$ Pseudopods", title_fontsize=8)
#
#    ax2.set_title(r"(b) Decision Time $T_D$ [s]", fontsize=9, loc="left")
#    ax2.set_xlabel(r"Gradient $\nabla c$ [nM/\textmu m]")
#
#    for marker, num_pods in zip("osDv^", [2, 3, 6, 12]):
#        data = results[num_pods]
#        y = [np.mean(data[(g, 75.0)]["duration"]) for g in grads]
#        ax2.plot(grads, y, ls="--", lw=0.5, zorder=3, color="black")
#        ax2.errorbar(
#            grads,
#            y,
#            ls="",
#            marker=marker,
#            markersize=4,
#            capsize=1.5,
#            elinewidth=1,
#            color="black",
#            mfc="w",
#            mew=0.8,
#            zorder=3,
#            label=rf"${num_pods}$",
#        )
#
#    gc = bx1.get_gridspec()
#    bx1.remove()
#    bx2.remove()
#    bx1 = fig.add_subplot(gc[:, 1])
#    num_pods = list(results.keys())
##    for n in num_pods:
##        # Get the last experiment
##        y = results[n][experiments[-1]]["ci"]
##        #x = np.random.randn(len(y)) * 0.3
##        bx1.hist(y, bins=30, histtype='step')
#
##    power_scale_fit = []
##    calc_efficiency = lambda n, exp: (
##        results[n][exp]["ci"][(results[n][exp]["duration"] > 0)]
##        #/ results[n][exp]["duration"][results[n][exp]["duration"] > 0]
##    )
##    for exp in experiments:
##        y_eff = np.array([np.mean(calc_efficiency(n, exp)) for n in num_pods])
##        yerr_eff = np.array([np.std(calc_efficiency(n, exp), ddof=1) for n in num_pods])
##        fun = lambda x, a, b, c: a * x**-b + c
##        #popt, *_ = curve_fit(fun, num_pods, y_eff, p0=[1, 4, 0])
##        #power_scale_fit.append(popt[1])
##        if exp == list(experiments)[-1]:
##            bx1.errorbar(num_pods, y_eff, yerr=yerr_eff, fmt=".", color="black", capsize=2)
##            x = np.linspace(2, 12, 100)
##            #bx1.plot(x, fun(x, *popt), color="black", linestyle="--", lw=1.2, zorder=2)
##
##    bx1.set_xticks([2, 4, 6, 8, 10, 12])
##    bx1.set_xlabel("Number of Pseudopods")
##    bx1.set_title(r"(c) Decision Efficiency $\Xi$ [s$^{-1}$]", fontsize=9, loc="left")
##    exp = experiments[-1]
##    get_data = lambda n: (results[n][exp]["ci"][(results[n][exp]["duration"] > 0)], results[n][exp]["duration"][results[n][exp]["duration"] > 0])
##    for n in [2, 3, 6]:
##        ci, T = get_data(n)
##        bx1.plot(T, ci, rasterized=True, ls='', marker='.', mec='none', alpha=1)
##    bx1.set_xlim(0, 20)
##    bx1.set_xlabel(r"Decision Time $T_D$ [s]")
##    bx1.set_ylabel("Gradient Alignment")
#
##
##    # Add inste to bx1
##    bx2 = bx1.inset_axes(bounds=(0.45, 0.45, 0.45, 0.45))
##    bx2.plot(grads[1:], power_scale_fit[1:], ".", color="black")
#
#    plt.tight_layout()
#    figure_name = "speed_accuracy_decision_combined.pdf"
#    fig.savefig(figures_dir / figure_name, bbox_inches="tight")
#    print("Figure saved at:", figures_dir / figure_name)
#    plt.close(fig)
