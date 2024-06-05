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
from jax import jacfwd, jacrev

import os

from jax_pigp.utility_fns import softplus
from jax_pigp.datasets import TrainDataLinearPDE, TrainDataNonLinearPDE, TestData

from jax_pigp.base import Array
from typing import Callable


def jac_hess(funct: Callable, argnum: int):
    r"""
    Returns Jacobian and Hessian functions for input Callable funct,
    with respect to its argnum arghument
    """
    # callable Jacobian function
    jacobian = jacrev(funct, argnums=argnum)
    # callable Hessian function
    hessian = jacfwd(jacobian, argnums=argnum)
    return jacobian, hessian


def print_param_values(params: dict, opt_options: dict):
    r"""
    Print non-variational parameters to console assuming
    softplus transformation
    """
    for k, p in params.items():
        if k in ["a_whitened", "A_whitened_vec"]:
            continue
        print(f"{k}: {softplus(p)}")


def load_test_data(data_path: str):
    r"""
    Load test inputs Xs and outputs us from directory at data_path
    """
    Xs = jnp.load(f"{data_path}/Xs.npy")
    us = jnp.load(f"{data_path}/us.npy")
    return TestData(Xs, us)


# function to check if a directory exists at given path
dir_exists = lambda dir_path: os.path.isdir(dir_path)

# function to check if a file exists at given path
file_exists = lambda file_path: os.path.isfile(file_path)

def load_data(
    data_path: str, Nu: int = 0, Nf: int = 0, Ng: int = 0, load_test: bool = True
):
    r"""
    Load training and test data for a linear PDE
    """

    if not dir_exists(data_path):
        raise NotADirectoryError(f"No directory at: {data_path}")

    def load_one_space(Xtrain: list, ytrain: list, space_label: str, N: int):
        r"""
        Load first N inputs/outputs from function space specified by space_label
        """

        if file_exists(f"{data_path}/X{space_label}.npy") and file_exists(
            f"{data_path}/y{space_label}.npy"
        ):
            X = jnp.load(f"{data_path}/X{space_label}.npy")[:N]
            y = jnp.load(f"{data_path}/y{space_label}.npy")[:N]
            Xtrain.append(X)
            ytrain.append(y)
        else:
            if N > 0:
                print(
                    f"Warning: Requested N{space_label}={N} but X{space_label}.npy and/or y{space_label}.npy are not in {data_path}"
                )

        return Xtrain, ytrain

    # lists to hold the training inputs/outputs from each space (u/f/g space)
    Xtrain = []
    ytrain = []

    # load u (function) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "u", Nu)

    # load f (PDE) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "f", Nf)

    # load g (ISC) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "g", Ng)

    assert len(Xtrain) > 0, f"No data was loaded from {data_path}"

    train_data = TrainDataLinearPDE(Xtrain, ytrain)

    if not load_test:
        return train_data

    test_data = load_test_data(data_path)

    return train_data, test_data


def load_data_nonlinear(
    data_path: str, Nu: int = 0, Nf: int = 0, Ng: int = 0, load_test: bool = True
):
    r"""
    Load training and test data for a nonlinear PDE
    """

    if not dir_exists(data_path):
        raise NotADirectoryError(f"No directory at: {data_path}")

    def load_one_space(Xtrain: list, ytrain: list, space_label: str, N: int):
        r"""
        Load first N inputs/outputs from function space specified by space_label
        """

        if file_exists(f"{data_path}/X{space_label}.npy") and file_exists(
            f"{data_path}/y{space_label}.npy"
        ):
            X = jnp.load(f"{data_path}/X{space_label}.npy")[:N]
            y = jnp.load(f"{data_path}/y{space_label}.npy")[:N]
            Xtrain.append(X)
        else:
            # if no data loaded from this space, we set the output value to None for referencing inside the the NonlinearPSGP class
            y = None
            if N > 0:
                print(
                    f"Warning: Requested N{space_label}={N} but X{space_label}.npy and/or y{space_label}.npy are not in {data_path}"
                )
        ytrain.append(y)

        return Xtrain, ytrain

    # lists to hold the training inputs/outputs from each space (u/f/g space)
    Xtrain = []
    ytrain = []

    # load u (function) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "u", Nu)

    # load f (PDE) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "f", Nf)

    # load g (ISC) space data
    Xtrain, ytrain = load_one_space(Xtrain, ytrain, "g", Ng)

    assert len(Xtrain) > 0, f"No data was loaded from {data_path}"

    train_data = TrainDataNonLinearPDE(Xtrain, ytrain, [Nu, Nf, Ng])

    if not load_test:
        return train_data

    test_data = load_test_data(data_path)

    return train_data, test_data



## Plotting functionality

ONE_COLUMN_FIGSIZE = (4, 3)
TWO_COLUMN_FIGSIZE = (7, 3)
THREE_COLUMN_FIGSIZE = (12, 3)


def make_heatmap(fig, ax, output_grid, test_data, Xu=None, Xf=None, title=None):

    xlb, tlb = test_data.Xs.min(0)
    xub, tub = test_data.Xs.max(0)

    h = ax.imshow(
        output_grid,
        interpolation="nearest",
        cmap="rainbow",
        extent=[tlb, tub, xlb, xub],
        origin="lower",
        aspect="auto",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)

    cbar.ax.tick_params(labelsize=15)

    if Xu is not None:
        ax.scatter(Xu[:, 1], Xu[:, 0], alpha=0.5, s=10, c="k")
    if Xf is not None:
        ax.scatter(Xf[:, 1], Xf[:, 0], alpha=0.5, s=10, c="k", marker="x")

    ax.set_xlabel("t")
    ax.set_ylabel("x")

    if title is not None:
        ax.set_title(title)


def plot_lc(ax, values_arr, title=None):
    ax.plot(values_arr)
    if title is not None:
        ax.set_title(title)
