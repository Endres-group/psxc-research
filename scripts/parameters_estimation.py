"""
Let's find the best parameters for the model using bayex
"""
from pathlib import Path
import functools
from re import A
import jax
import jax.numpy as jnp
import itertools

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

import bayex

import psxc

current_dir = Path(__file__).resolve().parent
n_reps = 1000
x0 = 50.0
grads = [0.01, 0.02, 0.05, 0.1, 0.25, 0.75, 1.4, 1.8]
values = [5, 25.0, 75.0, 125.0, 175.0]
experiments = list(itertools.product(grads, values))


# Load the datapoints, and fit a curve to the data
exp_points = np.loadtxt(current_dir / "data/fig1c_scirep.dat", skiprows=1, delimiter=",")
f = lambda snr, a, b: psxc.perfect_absorber_limit(10**snr, a=a, b=b)
(a, b), _ = curve_fit(f, exp_points[:, 0], exp_points[:, 1], sigma=exp_points[:, 2], bounds=([0, 0], [1, 1]))
target = a

def get_ci_and_snr(key, params, grad, value):

    signal_fn = lambda _: jnp.ones(12)
    profile = psxc.LinearProfile(grad, value, x=x0)
    splitting_event = functools.partial(
        psxc.run_chemotactic_trajectory, signal_fn=signal_fn, num_steps=20, warmup=10, t_max=100, dt=0.1
    )

    collapse = lambda x: x.reshape(-1, *x.shape[2:])

    def _single_run(key, x0):
        key, key_cell = jax.random.split(key)
        cell_state = psxc.CellState.spawn(key_cell, x=x0, length=0.5)
        xs, mask = splitting_event(params, key, cell_state, profile)
        xs, mask = jax.tree_util.tree_map(collapse, (xs, mask))
        ci = psxc.weighted_chemotactic_index(xs)
        snr = psxc.signal_to_noise(xs, profile)
        return ci, snr, mask

    x0s = jax.random.normal(key, (n_reps,)) * 20 + x0
    keys = jax.random.split(key, n_reps)
    outs = jax.vmap(_single_run)(keys, x0s)
    return jax.tree_util.tree_map(collapse, outs)

def to_numpy(x):
    return jax.tree.map(np.asarray, x)

# Calculate SNR
def objective_function(activation_rate, noise_strength):
    params = psxc.Parameters(activation_rate=1/activation_rate, noise_strength=noise_strength)

    key = jax.random.key(0)

    results = {}
    for i, (grad, value) in enumerate(experiments):
        key_exp = jax.random.fold_in(key, i)
        ci, snr, mask = to_numpy(get_ci_and_snr(key_exp, params, grad, value))
        results[(grad, value)] = {"ci": ci[mask], "snr": snr[mask]}

    sizes = {e: len(results[e]["snr"]) for e in results}
    total_size = sum(sizes.values())

    SNR = np.empty(total_size)
    CI = np.empty(total_size)

    i = 0
    for experiment in results:
        n = sizes[experiment]
        SNR[i : i + n] = results[experiment]["snr"]
        CI[i : i + n] = results[experiment]["ci"]
        i += n

    valid = SNR < 0
    x, y, yerr = psxc.put_on_bins(SNR[valid], CI[valid], bins=np.linspace(-6.5, 0, 30))

    pal_to_fit = lambda snr, a, b: psxc.perfect_absorber_limit(10**snr, a=a, b=b)
    try:
        fit_params, *_ = curve_fit(pal_to_fit, x, y, p0=[0.001, 0.8], bounds=[0, 2], nan_policy='omit', sigma=yerr)
        loss = 100 * (fit_params[0] - target)**2
    except:
        loss = 1e3
    return -loss

# Run the optimization
#domain = {'activation_rate': bayex.domain.Real(1e-3, 8.0),
#          'noise_strength': bayex.domain.Real(1e-5, 1e-1)}
#
#optimizer = bayex.Optimizer(domain=domain, maximize=False, acq='PI')
#
## Define some prior parameters
#prior_params = {'activation_rate': [1/5, 1/0.5, 1],
#                'noise_strength': [1.5e-3, 1.5e-3, 1.5e-3]}
#ys = [objective_function(x1, x2) for x1, x2 in zip(prior_params['activation_rate'], prior_params['noise_strength'])]
#opt_state = optimizer.init(ys, prior_params)
#print(ys)
#
## Sample new points using Jax PRNG approach.
#ori_key = jax.random.key(42)
#for step in range(100):
#    key = jax.random.fold_in(ori_key, step)
#    new_params = optimizer.sample(key, opt_state)
#    y_new = objective_function(**new_params)
#    print(f'loss: {y_new}', new_params)
#    opt_state = optimizer.fit(opt_state, y_new, new_params)
#
#breakpoint()
## Print the best parameters
#arg = np.argmin(opt_state['ys'])
#print('Best parameters:', opt_state['params'][arg])

import bayes_opt

# bounded region
pbounds = {'activation_rate': (1e-3, 8.0), 'noise_strength': (1e-5, 1e-1)}

optimizer = bayes_opt.BayesianOptimization(f=objective_function, pbounds=pbounds, verbose=2)
optimizer.maximize(init_points=20, n_iter=100)

