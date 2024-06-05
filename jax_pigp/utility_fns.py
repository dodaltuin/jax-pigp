"""
File: utility_fns.py
Description: various utility functionality
"""

import jax.numpy as jnp
from jax import random

from tensorflow_probability.substrates import jax as tfp

triangle_transform = tfp.bijectors.FillScaleTriL(diag_shift=None)

from .base import FloatValue, Array


def add_noise(cov_matrix: Array, noise_vars: Array, indicator_var: Array) -> Array:
    r"""
    Adds elements of noise_vars to diagonal of cov_matrix, according to
    the values in indicator_var
    """
    return cov_matrix + jnp.diag(noise_vars[indicator_var])


def softplus(x: Array, inv: bool = False) -> Array:
    r"""
    Implements softplus function on input arg x. The arg inv specifies whether
    to consider inverse of softplus or not. Setting inv=True is useful when
    initialising parameters based on their softplus transformed value
    """
    if not inv:
        return jnp.log(1 + jnp.exp(x))
    else:
        return jnp.log(jnp.exp(x) - 1)


# default settings for generation of noise/kernel/pde parameters
PARAM_OPTIONS_DICT = {}
PARAM_OPTIONS_DICT["ls_bounds"] = [0.5, 5.0]
PARAM_OPTIONS_DICT["amp_bounds"] = [5.0, 25.0]
PARAM_OPTIONS_DICT["alpha_bounds"] = [5.0, 25.0]
PARAM_OPTIONS_DICT["noise_std_init"] = 0.1
PARAM_OPTIONS_DICT["noise_transform"] = softplus
PARAM_OPTIONS_DICT["theta_transform"] = lambda x, inv: x


def generate_params(
    prng_key: int, param_options_dict: dict = PARAM_OPTIONS_DICT
) -> dict:

    # initialise dictionary to hold parameter values
    params = {}

    s0, s1, s2 = random.split(prng_key, 3)

    # dimensionality of the input space (including any temporal component)
    input_dim = (
        param_options_dict["input_dim"] if "input_dim" in param_options_dict else 1
    )

    # initialise kernel parameters
    amp_min, amp_max = param_options_dict["amp_bounds"]
    params[f"amp"] = softplus(
        random.uniform(s0, minval=amp_min, maxval=amp_max, shape=(1,)), True
    )

    ls_min, ls_max = param_options_dict["ls_bounds"]
    params[f"ls"] = softplus(
        random.uniform(s1, minval=ls_min, maxval=ls_max, shape=(input_dim,)), True
    )

    alpha_min, alpha_max = param_options_dict["alpha_bounds"]
    params[f"alpha"] = softplus(
        random.uniform(s2, minval=alpha_min, maxval=alpha_max, shape=(1,)), True
    )

    # initialise three noise levels (noise in u-space, noise in f-space, and noise in g-space)
    noise_trans = param_options_dict["noise_transform"]
    noise_init = param_options_dict["noise_std_init"]
    params[f"noise_std"] = noise_trans(jnp.array(noise_init), True)

    # initialise PDE parameters theta, if considering inverse problem
    if "theta_init" in param_options_dict:
        theta_trans = param_options_dict["theta_transform"]
        params[f"theta"] = theta_trans(param_options_dict["theta_init"], True)

    # initialise (whitened) variational mean/covariance parameters a/A, if considering nonlinear PDE
    if "nonlinear" in param_options_dict:
        assert (
            "Nh" in param_options_dict
        ), "If using nonlinear PIGP, must specify the dimensionality of the latent space when initialising the variational parameters"
        Nh = param_options_dict["Nh"]
        params["a_whitened"] = jnp.zeros((Nh, 1), dtype=jnp.float64)
        params["A_whitened_vec"] = triangle_transform.inverse(
            jnp.identity(Nh, dtype=jnp.float64)
        )

    return params
