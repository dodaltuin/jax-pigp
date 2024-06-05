"""
File: expectations.py
Description: functionality to compute the terms in the ELBO involving expected values
of the observational data
"""

from jax import random, vmap
from jax.random import multivariate_normal
from jax.scipy.stats.norm import logpdf as log_norm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as log_mvn_pdf

from typing import Callable

from ..base import FloatValue, Array

# random seed used when performing Monte-Carlo sampling in estimate log_prob_yf
RANDOM_SAMPLING_KEY = random.PRNGKey(1)


def compute_exp_log_prob(
    obs: FloatValue, obs_mean: FloatValue, obs_var: FloatValue, noise_std: FloatValue
):
    r"""
    Computes expected value of the log of a Gaussian pdf

    Used to compute terms involving u/g-space observations in the ELBO
    """

    return log_norm_pdf(obs, obs_mean, noise_std) - 0.5 * obs_var / noise_std**2


# vmap so that function can be called across different data points for fixed noise level
compute_exp_log_prob_vmap = vmap(compute_exp_log_prob, in_axes=[0, 0, 0, None])

# vmap so that log pdf value can be evalauted across different samples of the mean vector
# given fixed observation value and fixed noise level
log_mvn_pdf_vmap = vmap(log_mvn_pdf, [None, 0, None])


def estimate_log_prob_yf(
    params: dict,
    mean_vector: Array,
    cov_matrix: Array,
    yf: Array,
    sigma2_f: FloatValue,
    F_vmap: Callable,
    len_d: int,
    n_samples: int,
):
    r"""
    Evaluates Monte-Carlo esimate of the expected value of the log of a Gaussian
    pdf under non-linear transformation F_vmap

    Used to compute term involving f-space observations yf in the ELBO
    """

    # generate n_samples random samples and reshape so that result can be passed to F_vmap
    random_samples = multivariate_normal(
        RANDOM_SAMPLING_KEY, mean_vector[:, 0], cov_matrix, shape=(n_samples,)
    ).reshape((n_samples, len_d, yf.shape[0]))

    # evaluate algebraic representation of the nonlinear PDE (F_vmap) on each random sample
    F_samples = F_vmap(params, random_samples)

    # each sample in F_samples used to generate a value of log_pyf
    log_pyf_samples = log_mvn_pdf_vmap(yf[:, 0], F_samples, sigma2_f)

    # estimate the expected value as the mean of the Monte-Carlo samples
    exp_log_pyf_estimate = log_pyf_samples.mean()

    return exp_log_pyf_estimate
