import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit
from tqdm import tqdm

import chemotactic_index as ci_module
import psxc
CURRENT_FILE = pathlib.Path(__file__)
CURRENT_DIR = CURRENT_FILE.parent
sys.path.append(str(CURRENT_DIR))

# Load the weights from the last run
RUNS_DIR = CURRENT_DIR.parent / "runs"
ci_module.N_REPS = 2_000

weights, discrete = psxc.utils.load_last_model(RUNS_DIR)
model = psxc.ppo.ActorCritic(12, discrete=discrete)

ENVS = ci_module.ENVS

DETERMINSITIC = False

def ppo_signal(snr, deterministic=DETERMINSITIC):
    # Make it a vector
    key = jax.random.fold_in(jax.random.PRNGKey(42), 1000 * snr)
    snr = jnp.array([snr])
    logits = model.apply(weights, snr)[1]
    if deterministic:
        action = jnp.argmax(jax.nn.softmax(logits, axis=-1), -1)
    else:
        action = jax.random.categorical(key, logits, axis=-1)
    return action

def load_configuration(path):
    with open(path, "r") as f:
        return json.load(f)

if __name__ == '__main__':

    # Load the configuration
    configs = load_configuration(CURRENT_DIR / "data/ci_results.json")

    # PPO Config
    ppo_config = {"signal_fn": ppo_signal, "label": "PPO", "color": "green", "linestyle": "-"}


    plotting_data = {}

    logistic_fn = lambda x, a, b, c: a / (1 + np.exp(-b * (x - c)))

    # Add the plots of the other configurations
    for name, config in configs.items():
        popt = config["popt"]
        x_fit = np.linspace(-6.5, 0, 1000, endpoint=False)
        y_fit = logistic_fn(x_fit, *popt)
        plotting_data[name] = {"x": x_fit, "y": y_fit}

    metrics = ["ci", "snr", "msd", "time"]
    results = {}
    key = jax.random.PRNGKey(42)
    for n_start, (grad, value) in enumerate(tqdm(ENVS, ncols=80)):
        key_exp = jax.random.fold_in(key, n_start)
        ci, snr, mask, d, t = ci_module.to_numpy(ci_module.get_ci_and_snr(key_exp, grad, value, ppo_signal))
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
    popt, *_ = curve_fit(logistic_fn, x, y, sigma=yerr, p0=[0.8, 1.0, -5])
    print("Fit parameters:", popt)

    x_fit = np.linspace(-6.5, 0, 1000, endpoint=False)
    y_fit = logistic_fn(x_fit, *popt)
    plotting_data["PPO"] = {"x": x, "y": y, "yerr": yerr, "x_fit": x_fit, "y_fit": y_fit, "popt": popt}


    x_theory = np.linspace(-6.5, 0, 1000, endpoint=False)
    y_theory = 0.9 * psxc.perfect_absorber_limit(10**x_theory, a=50)

    # Plot the results
    FIGURES_DIR = CURRENT_DIR / pathlib.Path("figures")

    plt.style.use(["science"])

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0), dpi=200)
    ax.set_xlabel(r"Signal-to-Noise $\log_{10}$(SNR) [nM/\textmu m]")
    ax.set_ylabel(r"Chemotactic Index")

    ax.plot(x_theory, y_theory, ls="-", color="black", lw=1, alpha=0.5)

    for name, data in plotting_data.items():
        if name in configs:
            x, y = data["x"], data["y"]
            color = configs[name]["color"]
            label = configs[name]["label"]
            ax.plot(x, y, ls="-", color=color, lw=1.5, alpha=0.5)
        elif name in "PPO":
            x, y = data["x_fit"], data["y_fit"]
            color = ppo_config["color"]
            label = ppo_config["label"]
            ax.errorbar(data["x"], data["y"], yerr=data["yerr"], ls="none", color=color, ms=3, marker='o', mfc=color, zorder=2, mew=0.0)
            ax.plot(x, y, ls="-", color=color, lw=2, label=label, zorder=4)

    ax.set_xlim(-6.5, 0)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(-6, 1, 1), [fr"$10^{{{i}}}$" for i in np.arange(-6, 1, 1)])

    # Plot about which pseudopods are active in each setup as insets
    for ((pos_x, pos_y), snr) in zip(((0.05, 0.45), (0.6, 0.2)), (-5, -1)):
        lax = ax.inset_axes([pos_x, pos_y, 0.3, 0.3])
        lax.axis("off")
        lax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
        theta = np.linspace(0, 2 * np.pi, 13)[:-1] + np.pi / 2
        x = np.cos(theta)
        y = np.sin(theta)
        #lax.plot(x, y, "o", color="white", mec="black", markersize=7)
        for i in range(len(x)):
            lax.plot([0, x[i]], [0, y[i]], color="black", lw=5, solid_capstyle="round", alpha=0.2)
        ind = np.argwhere(ppo_signal(snr, deterministic=True))[...,0]
        print(ind)
        x_active = x[ind]
        y_active = y[ind]
        print(x_active)
        #lax.plot(x_active, y_active, "o", color="green", markersize=7, mec="black")
        for i in range(len(x_active)):
            lax.plot([0, x_active[i]], [0, y_active[i]], color="green", lw=5, solid_capstyle="round")
        lax.set_aspect("equal")

    legend = ax.legend(fontsize=8)
    plt.tight_layout()

    # Save the figure
    figname = "chemotactic_index_with_ppo.pdf"
    fig.savefig(FIGURES_DIR / figname, bbox_inches="tight")

    # Plot the mapping from SNR to strengths
    fake_input = jnp.linspace(-6, 2, 8)
    fake_output = jax.nn.softmax(model.apply(weights, fake_input[:, None])[1], axis=-1)
    fake_output = jnp.array(fake_output)[..., 1]
    #fake_output = jnp.argmax(fake_output, axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 3.0), dpi=200)
    ax.set_title("Pseudopod Strengths", fontsize=10, loc="left")
    ax.set_xlabel(r"Signal-to-Noise $\log_{10}$(SNR) [nM/\textmu m]")
    ax.set_xticks(np.arange(-6, 1, 1), [fr"$10^{{{i}}}$" for i in np.arange(-6, 1, 1)])
    pcm = ax.pcolormesh(fake_input, np.arange(12), fake_output.T, cmap="magma", lw=0, rasterized=True)
    fig.colorbar(pcm, ax=ax)
    fig.savefig(FIGURES_DIR / "ppo_mapping.pdf", bbox_inches="tight")

