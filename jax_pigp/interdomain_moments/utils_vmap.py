"""
File: utils_vmap.py
Description: functionality for vmapping mean and kernel functions
"""

from jax import vmap

from ..base import IntegerList, Callable, CallableList


def vmap_mean_fn(m_fn: Callable) -> Callable:
    """
    vmap a mean function to allow it to be applied over array
    of input data points. Note the first argument is assumed
    to be the parameter dictionary which is not vmapped over
    """
    return vmap(m_fn, (None, 0))


def vmap_kernel_fn(k_fn: Callable) -> Callable:
    """
    vmap a kernel function to allow it to be applied pairwise
    over two arrays of input data points. Again, the first
    argument is assumed to be the parameter dictionary, which
    is not vmapped over
    """
    return vmap(vmap(k_fn, (None, None, 0)), (None, 0, None))


def check_shapes(linear_operators: CallableList, Xtrain_indices: IntegerList) -> None:
    """
    Check consistency between the number of linear operators and the number of Xtrain indices
    """
    if Xtrain_indices is None:
        return

    assert len(linear_operators) == len(Xtrain_indices)
