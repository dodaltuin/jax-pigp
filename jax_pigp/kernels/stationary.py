"""
File: stationary.py
Description: Implements stationary kernel functions
"""

import jax.numpy as jnp

from ..utility_fns import softplus
from ..base import Array, FloatValue


def squared_exponential(params: dict, loc1: Array, loc2: Array) -> FloatValue:
    r"""
    Evaluates squared-exponential kernel on pair of inputs loc1 and loc2 from the
    spatio-temporal domain, with kernel hyper-parameters stored in params
    """

    # extract kernel parameter values
    amp = softplus(params["amp"])
    ls = softplus(params["ls"])

    # evaluate squared-exponential kernel at inputs loc1, loc2
    cov_val = amp * jnp.exp(-0.5 * jnp.sum(((loc1 - loc2) / ls) ** 2, axis=-1))

    return cov_val.squeeze()


def rational_quadratic(params: dict, loc1: Array, loc2: Array) -> FloatValue:
    r"""
    Evaluates rational-quadratic kernel on pair of inputs loc1 and loc2 from the
    spatio-temporal domain, with kernel hyper-parameters stored in params
    """

    # extract kernel parameter values
    amp = softplus(params["amp"])
    ls = softplus(params["ls"])
    alpha = softplus(params["alpha"])

    # evaluate rational-quadratic kernel at inputs loc1, loc2
    cov_val = (amp) * (
        1.0 + jnp.sum(((loc1 - loc2) / ls) ** 2, axis=-1) / (2.0 * alpha)
    ) ** (-alpha)

    return cov_val.squeeze()
