"""
This script computes metrics on the decision making dynamics of actin polymerization.

Author: Albert Alonso
Date: 2024.05.10
"""
import itertools
import pathlib

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

import scienceplots

import psxc

matplotlib.use("Agg")

current_dir = pathlib.Path(__file__).parent
figures_dir = current_dir / 'figures'

# Parameters 
grads = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.4]
#grads = np.logspace(-2, 0.5, 8)
values = [5, 25.0, 75.0, 125.0, 175.0]
experiments = list(itertools.product(grads, values))

GRAD_TRAJECTORY = 0.01
VALUE_TRAJECTORY = 125.0

length = 1.0
params = psxc.Parameters(cross_inhibition=0.5)


@jax.jit
def compute_time(key, grad, value, x0=50, num_reps=int(1e4), dt=0.05):
    profile = psxc.LinearProfile(grad, value, x=x0)

    def pseudopods_lengths(xs):
        diff = xs - xs[:, -2, jnp.newaxis]
        return jnp.linalg.norm(diff, axis=-1)

    def calculate_spread(xs):
        """ Calculate distance between pseudopod tips"""
        diff = xs - xs[:, -2, jnp.newaxis]
        lengths = jnp.linalg.norm(diff, axis=-1)
        # Calculate the distance of the oo
        #D = jnp.max(jnp.sqrt(jnp.sum((xs[:, :-2][:, None] - xs[:, :-2][None])**2, axis=-1)), axis=(1, 2))
        return D

    def decision_time(key):
        key, key_cell = jax.random.split(key)
        cell_state = psxc.CellState.spawn(key_cell, x=x0, length=length)
        (xs, mask), xf, T = psxc.sensing_event(params, key, cell_state, profile, t_max=40, dt=dt)
        decision_time, growth_time = psxc.decision_time(xs, mask, alpha=0.01, dt=dt)
        lengths = pseudopods_lengths(xs)
        num_pseudopods = jnp.sum(jnp.cumsum(lengths > 0.1, axis=0) > 50, axis=1).max()
        #max_spread = calculate_spread(xs)
        pseudopod_count = jnp.sum(lengths[(decision_time/dt).astype(int)] > 0.05)
        #pseudopod_count = jnp.mean(lengths, axis=(0,1), where=mask[:, None])
        results = {'Td': decision_time, 'Tg': growth_time, 'T': T, 'nps': pseudopod_count}
        return results

    keys = jax.random.split(key, num_reps)
    results = jax.vmap(decision_time)(keys)
    return results


key = jax.random.key(42)
results = {}
for i, (grad, val) in enumerate(tqdm(experiments, ncols=81)):
    key_exp = jax.random.fold_in(key, i)
    results[(grad, val)] = jax.tree.map(np.asarray, compute_time(key_exp, grad, val))


@jax.jit
def get_lengths_and_actin(key, grad, value, x0=10, dt=0.01):

    params = psxc.Parameters()

    def pseudopods_lengths(xs):
        diff = xs - xs[:, -2, jnp.newaxis]
        return jnp.linalg.norm(diff, axis=-1)

    profile = psxc.LinearProfile(grad, value, x=x0)
    key, key_cell = jax.random.split(key)
    cell_state = psxc.CellState.spawn(key_cell, x=x0, length=length)
    (xs, mask), xf, T, ys = psxc.sensing_event(params, key, cell_state, profile, t_max=40, dt=dt, has_aux=True)
    lengths = pseudopods_lengths(xs)
    times = jnp.cumsum(mask) * dt
    decision_time, growth_time = psxc.decision_time(xs, mask, alpha=0.1, dt=dt)
    return {'l': lengths, 't': times, 'actin': ys, 'Td': decision_time, 'Tg': growth_time, 'T': T, 'mask': mask}

actin_results = {}
for GRAD, VALUE in zip([0.01, 1.0], [50, 150]):
    res = jax.tree.map(np.asarray, get_lengths_and_actin(key, GRAD, VALUE))
    res['actin'] = res['actin'][res['mask']]
    res['t'] = res['t'][res['mask']]
    res['l'] = res['l'][res['mask']]
    actin_results[(GRAD, VALUE)] = res

# Plotting
plt.style.use('science')
fig, axes = plt.subplot_mosaic([['a', 'a'], ['b', 'b'], ['c', 'd']], figsize=(5.5, 6), dpi=300, layout='tight', width_ratios=[1.5, 1])

# Subplot (a) and (b)
for label, ((grad, value), res) in zip('ab', actin_results.items()):
    if label == 'b':
        axes[label].set_xlabel(r'Time [s]')
    axes[label].set_ylabel(r'$A_i$')
    colors = plt.cm.jet(np.linspace(0.0, 1, 12, endpoint=False))
    if label == 'a':
        inset = axes[label].inset_axes([0.4, 0.6, 0.3, 0.3], zorder=3)
        for i in range(12):
            thetas = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            x = np.cos(thetas[i])
            y = np.sin(thetas[i])
            inset.scatter(x, y, color=colors[i], s=50)
            inset.set_aspect('equal')
            inset.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
            inset.axis('off')

    for pod in range(res['actin'].shape[1]):
        l = res['actin'][:, pod]
        t = res['t']
        color = colors[pod] if pod < 12 else 'black'
        axes[label].plot(t, l, color=color, lw=0.8, zorder=2)

    # Color half to the left in one color and half to the right in other
    axes[label].set_ylim(0, 1)
    axes[label].set_xlim(0, res['t'][-1])
    axes[label].set_yticks([0, 0.5, 1])
    axes[label].axvline(res['Td'], color='black', linestyle='-', lw=1.0)
    axes[label].text(res['Td'], 0.7, r'$T_D$', ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', edgecolor='none'))
    #axes[label].fill_betweenx([-1, 2], -2, dynamics_results['Td'], color='gainsboro', alpha=0.5)
    #axes[label].text(0.3, 0.9, r'Decision-making', transform=axes[label].transAxes, ha='center', va='center', fontsize=10, color='black')
    axes[label].text(0.3, 0.9, rf'$\nabla c = {grad}$ nM/\textmu m', transform=axes[label].transAxes, ha='center', va='center', fontsize=10, color='black')
    axes[label].text(0.3, 0.8, rf'$c = {value}$ nM', transform=axes[label].transAxes, ha='center', va='center', fontsize=10, color='black')
    #axes[label].fill_betweenx([-1, 2], dynamics_results['Td'], 20, color='lightsteelblue', alpha=0.5)
    #axes[label].text(0.85, 0.9, r'Navigation', transform=axes[label].transAxes, ha='center', va='center', fontsize=10, color='black')
    axes[label].set_ylim(0, 1)
    axes[label].set_xlim(0, res['t'][-1])
    axes[label].set_yticks([0, 0.5, 1])

# Subplot (b) Number of pseudopods as a function of gradient
#axes['b'].set_ylabel(r'Length')
#colors = plt.cm.Greens(np.linspace(0.5, 1, 12))
#for pod in range(dynamics_results['l'].shape[1]):
#    l = dynamics_results['l'][:, pod]
#    t = dynamics_results['t']
#    color = colors[pod] if pod < 12 else 'gray'
#    axes['b'].plot(t, l, color=color, lw=1.0, zorder=4)
#
## Color half to the left in one color and half to the right in other
#axes['b'].axvline(dynamics_results['Td'], color='black', linestyle='--', lw=1.0)
#axes['b'].text(dynamics_results['Td'], 0.7, r'$T_D$', ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', edgecolor='none'))
##axes['b'].fill_betweenx([-1, 2], -2, dynamics_results['Td'], color='gainsboro', alpha=0.5)
#axes['b'].text(0.4, 0.9, r'Decision-making', transform=axes['b'].transAxes, ha='center', va='center', fontsize=8, color='black')
##axes['b'].fill_betweenx([-1, 2], dynamics_results['Td'], 20, color='lightsteelblue', alpha=0.5)
#axes['b'].text(0.85, 0.9, r'Navigation', transform=axes['b'].transAxes, ha='center', va='center', fontsize=8, color='black')
#axes['b'].set_ylim(0, 1)
#axes['b'].set_xlim(0, dynamics_results['t'][-1])
#axes['b'].set_yticks([0, 0.5, 1])
#for val, marker, color in zip(values, 'o^sDv', 'bgrcm'):
#    if val != 75:
#        continue
#    avg_pseudopods = [np.mean(results[(grad, val)]['nps']) for grad in grads]
#    axes['b'].plot(grads, avg_pseudopods, marker=marker, label=f'${val: >3g}$ nM', mfc='w', mec='k', ms=4, ls='', zorder=4)
#    break

#axes['b'].set_ylabel(r'$\langle N_{\mathcal{{P}}} \rangle$')
#axes['b'].set_xlabel(r'$\nabla c$ [nM/\textmu m]')
#axes['b'].set_xticks([0, 0.5, 1.0, 1.5])

# Subplot (c)
sizes = [len(results[(grad, val)]['Td']) for grad, val in experiments]
total_size = sum(sizes)
decision_times = np.empty(total_size)
growth_times = np.empty(total_size)
for i, (grad, val) in enumerate(experiments):
    start = sum(sizes[:i])
    end = sum(sizes[:i + 1])
    decision_times[start:end] = results[(grad, val)]['Td']
    growth_times[start:end] = results[(grad, val)]['Tg']
axes['c'].set_xlabel(r'Duration [s]')
bins = np.linspace(0, 20, 30)
axes['c'].hist(decision_times, bins=bins, color='gray', alpha=1.0, density=True, edgecolor='black', rwidth=1)
axes['c'].hist(decision_times+growth_times, bins=bins, color='gainsboro', alpha=0.8, density=True, edgecolor='black', rwidth=1)
f = gaussian_kde(decision_times)
x = np.linspace(0, 20, 1000)
axes['c'].plot(x, f(x), color='red', linestyle='-', label=r'$T_D$', lw=1.5)
f = gaussian_kde(decision_times+growth_times)
axes['c'].set_yticks([])
axes['c'].plot(x, f(x), color='blue', linestyle='-', label=r'$T$', lw=1.5)
axes['c'].legend()

# Subplot (d)
for val, marker, color in zip(values, 'o^sDv', 'bgrcm'):
    x = [grad for grad, value in experiments if value == val]
    y = [np.mean(results[(grad, val)]['Td']) for grad in x]
    axes['d'].plot(x, y, marker=marker, label=f'${val: >3g}$', mfc='w', mec='k', ms=4, ls='', zorder=4)
    f = lambda x, a, b, c: a * np.exp(-b * x) + c
    try:
        popt, _ = curve_fit(f, x, y, p0=[10, 0.5, 0])
        x = np.linspace(0, 1.5, 100)
        y = f(x, *popt)
        axes['d'].plot(x, y, color=color, ls='--', lw=1.0, zorder=3)
    except:
        pass
axes['d'].set_ylabel(r'$\langle T_D \rangle$ [s]')
axes['d'].set_xlabel(r'$\nabla c$ [nM/\textmu m]')
axes['d'].set_ylim(3.0, 10)
axes['d'].set_yticks([4, 6, 8])
axes['d'].set_xticks([0, 0.5, 1.0, 1.5])
axes['d'].legend(title=r'$c_0 [nM]$', loc='upper right', fontsize=8, title_fontsize=8, labelspacing=0.2, ncols=1, columnspacing=0.1)

# Add labels to the subplots
#real_labels = {'a': 'a', 'b': 'b', 'c': 'b', 'd': 'c'}
for label, ax in axes.items():
    #real_label = real_labels[label]
    ax.text(0.02, 0.98, fr'\textbf{{({label})}}', transform=ax.transAxes, weight='bold', va='top', ha='left', fontsize=11)
plt.tight_layout()
plt.savefig(figures_dir / 'decision_making.pdf', bbox_inches='tight', dpi=300)
