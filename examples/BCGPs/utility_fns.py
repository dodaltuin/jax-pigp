"""
File: utility_fns.py
Description: utility functions relating to differential operators, data loading
and plotting, used to perform the experiments in the notebooks in this directory
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import jax.numpy as jnp
from jax import jacfwd, jacrev, random

import os

from jax_pigp.utility_fns import softplus
from jax_pigp.datasets import TrainDataLinearPDE, TrainDataNonLinearPDE, TestData

from typing import Callable

def jac_hess(funct: Callable, argnum: int):
    jacobian = jacrev(funct,argnums=argnum)
    hessian  = jacfwd(jacobian,argnums=argnum)
    return jacobian,hessian

def print_param_values(params: dict, opt_options: dict):

    for k, p in params.items():
        if k[:6] == "kernel":
            print(k)
            for k2, p2 in p.items():
                print(f'{k2}: {softplus(p2)}')
            print(f'')

    noise_vals = params['noise_std']
    print(f'noise_std: {opt_options["noise_transform"](noise_vals)}')

    if "theta" in params:
        theta_val = params['theta']
        print(f'theta: {opt_options["theta_transform"](theta_val)}')



def load_test_data(data_path: str):

    Xs = jnp.load(f'{data_path}/Xs.npy')
    us = jnp.load(f'{data_path}/us.npy')

    return TestData(Xs,us)

dir_exists  = lambda dir_path: os.path.isdir(dir_path)
file_exists = lambda file_path: os.path.isfile(file_path)

def load_data(data_path: str, Nu: int = 0, Nf: int = 0, Nb: int = 0, load_test: bool =True):

    if not dir_exists(data_path):
        raise NotADirectoryError(f'No directory at: {data_path}')

    def load_one_space(Xtrain, ytrain, space_label, N):

        if (file_exists(f'{data_path}/X{space_label}.npy') and file_exists(f'{data_path}/y{space_label}.npy')):
            X = jnp.load(f'{data_path}/X{space_label}.npy')[:N]
            y = jnp.load(f'{data_path}/y{space_label}.npy')[:N]
            Xtrain.append(X)
            ytrain.append(y)
        else:
            if N>0:
                print(f'Warning: Requested N{space_label}={N} but X{space_label}.npy and/or y{space_label}.npy are not in {data_path}')

        return Xtrain, ytrain

    # lists to hold the training inputs/outputs from each space (u/f/g space)
    Xtrain = []
    ytrain = []

    # load u (function) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "u", Nu)

    # load f (PDE) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "f", Nf)

    # load boundary data (u space)
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "b", Nb)

    assert len(Xtrain)>0, f"No data was loaded from {data_path}"

    train_data = TrainDataLinearPDE(Xtrain, ytrain)

    if not load_test: return train_data

    test_data = load_test_data(data_path)

    return train_data, test_data
















ONE_COLUMN_FIGSIZE   = (4,3)
TWO_COLUMN_FIGSIZE   = (7,3)
THREE_COLUMN_FIGSIZE = (12,3)
TWO_BY_TWO_FIGSIZE = (10,6)

def make_heatmap(fig, ax, output_grid, test_data, Xu=None, Xf=None, title=None):

    xlb, tlb = test_data.Xs.min(0)
    xub, tub = test_data.Xs.max(0)

    h = ax.imshow(output_grid,
                  interpolation='nearest',
                  cmap='rainbow',
                  extent=[tlb, tub, xlb, xub],
                  origin='lower',
                  aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)

    cbar.ax.tick_params(labelsize=15)

    if Xu is not None: ax.scatter(Xu[:,1], Xu[:,0], alpha=0.5, s=10, c='k')
    if Xf is not None:  ax.scatter(Xf[:,1], Xf[:,0], alpha=0.5, s=10, c='k', marker='x')

    ax.set_xlabel('t')
    ax.set_ylabel('x')

    if title is not None: ax.set_title(title)

def plot_lc(ax, values_arr, title=None):
    ax.plot(values_arr)
    if title is not None: ax.set_title(title)

def make_scatter_heatmap(ax, X, output, title=None):
    plot_obj = ax.scatter(X[:,0], X[:,1], c=output, cmap="jet")
    ax.axis('equal')
    add_colorbar(plot_obj)
    if title is not None: ax.set_title(title)

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


PARAM_OPTIONS_DICT = {}

PARAM_OPTIONS_DICT['ls_bounds'] = [.5, 5.]
PARAM_OPTIONS_DICT['amp_bounds'] = [5., 25.]
PARAM_OPTIONS_DICT['alpha_bounds'] = [5., 25.]

PARAM_OPTIONS_DICT['noise_std_init'] = jnp.array([0.05]*4)

PARAM_OPTIONS_DICT['noise_transform'] = softplus
PARAM_OPTIONS_DICT['theta_transform'] = lambda x, inv: x

PARAM_OPTIONS_DICT['n_sub_kernels'] = 1

def generate_params(prng_key: int, param_options_dict: dict = PARAM_OPTIONS_DICT):

    # initialise dictionary to hold parameter values
    params = {}

    n_sub_kernels = param_options_dict['n_sub_kernels'] if "n_sub_kernels" in param_options_dict else 1

    kernel_input_dims = param_options_dict["input_dim"] if "input_dim" in param_options_dict else [1]*n_sub_kernels

    assert len(kernel_input_dims) == n_sub_kernels

    amp_min, amp_max     = param_options_dict['amp_bounds']
    ls_min, ls_max       = param_options_dict['ls_bounds']
    alpha_min, alpha_max = param_options_dict['alpha_bounds']

    for i, D in enumerate(kernel_input_dims):

        kernel_params_i = {}

        s0, s1, s2, prng_key = random.split(prng_key, 4)


        # initialise kernel parameters
        kernel_params_i[f'amp'] = softplus(random.uniform(s0,
                                                 minval=amp_min,
                                                 maxval=amp_max,
                                                 shape=(1,)), True)

        kernel_params_i[f'ls']  = softplus(random.uniform(s1,
                                                 minval=ls_min,
                                                 maxval=ls_max,
                                                 shape=(D,)), True)

        kernel_params_i[f'alpha'] = softplus(random.uniform(s2,
                                                            minval=alpha_min,
                                                            maxval=alpha_max,
                                                            shape=(1,)), True)

        params[f'kernel_params_{i+1}'] = kernel_params_i


    # initialise noise levels in each function space of interest
    noise_trans = param_options_dict['noise_transform']
    noise_init  = param_options_dict['noise_std_init']
    params[f'noise_std'] = noise_trans(noise_init, True)

    # initialise PDE parameters theta, if considering inverse problem
    if "theta_bounds" in param_options_dict:

        theta_trans = param_options_dict['theta_transform']
        theta_min, theta_max = param_options_dict['theta_bounds']
        params[f'theta'] = softplus(random.uniform(prng_key,
                                                   minval=theta_min,
                                                   maxval=theta_max,
                                                   shape=theta_min.shape), True)

    return params
