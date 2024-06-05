"""
File: kl_divergence.py
Description: Functionality for evaluating KL divergence
"""

import jax.numpy as jnp
from jax.scipy import linalg

from .base import FloatValue, Array


def gaussian_kl(m1: Array, L1: Array, m2: Array, K2: Array) -> FloatValue:
    r"""
    Computes KL divergence KL(p1||p2) where

        p1(x) = N(m1, L1@L1.T)
        p2(x) = N(m2, K2)

    ## Note: follows formula given here:
    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """

    K1 = L1 @ L1.T
    L2 = linalg.cholesky(K2, lower=True)

    # calculate quadratic form term
    m_delta = m1 - m2
    quad_right = linalg.solve_triangular(L2, m_delta, lower=True)
    quad_term = (quad_right**2).sum()

    # calculate trace term
    L2inv_K1 = linalg.solve_triangular(L2, K1, lower=True)
    K2inv_K1 = linalg.solve_triangular(L2.T, L2inv_K1, lower=False)
    trace_term = jnp.trace(K2inv_K1)

    # calculate log determinant term
    log_det_K1 = 2.0 * jnp.log(jnp.diag(L1)).sum()
    log_det_K2 = 2.0 * jnp.log(jnp.diag(L2)).sum()
    log_det_term = log_det_K2 - log_det_K1

    # dimensionality of data
    D = m1.shape[0]

    # KL divergence
    KL = 0.5 * (log_det_term - D + quad_term + trace_term)

    return KL
