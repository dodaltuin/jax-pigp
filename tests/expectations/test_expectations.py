"""
File: test_expecations.py
Description: Tests sample estimate from estimate_log_prob_yf() against
explicit value from compute_exp_log_prob_vmap() under identity transform
"""

# ensure that we can access jax_pigp
import sys
sys.path.append('../../')

from jax import config, random, scipy, vmap
import jax.numpy as jnp
config.update("jax_enable_x64", True)

from jax_pigp.expectations import compute_exp_log_prob_vmap, estimate_log_prob_yf

SAMPLE_ERROR_TOL = 0.001
N_SAMPLES        = 150000
# using the identity transformation means that the expected value is available in
# closed form, and can be evaluated using compute_exp_log_prob_vmap() above
F_vmap = vmap(lambda p, x: x)

def explicit_v_sample_estimate(obs, obs_mean, K, noise_std) -> None:
    """
    Check ...
    """
    obs_var = jnp.diag(K)
    exact_value = compute_exp_log_prob_vmap(obs, obs_mean, obs_var, noise_std).sum()
    sample_estimate = estimate_log_prob_yf(None, obs_mean, K, obs, noise_std**2, F_vmap, 1, N_SAMPLES)

    assert (jnp.abs(1 - sample_estimate/exact_value)) < SAMPLE_ERROR_TOL

def test_low_variance_low_noise() -> None:

    N = 50
    obs_mean = random.normal(random.PRNGKey(0), shape=(N,1))
    K = jnp.eye(N)*1e-4
    noise_std = 1e-6
    obs = obs_mean + random.normal(random.PRNGKey(55), shape=(N,1))*noise_std
    explicit_v_sample_estimate(obs, obs_mean, K, noise_std)

def test_low_variance_high_noise() -> None:

    N = 100
    obs_mean = random.normal(random.PRNGKey(10), shape=(N,1))
    K = jnp.eye(N)*1e-4
    noise_std = 1e-0
    obs = obs_mean + random.normal(random.PRNGKey(475), shape=(N,1))*noise_std
    explicit_v_sample_estimate(obs, obs_mean, K, noise_std)


def test_general_low_noise() -> None:

    N = 250
    obs_mean = random.normal(random.PRNGKey(0), shape=(N,1))
    L_init = jnp.tril(random.normal(random.PRNGKey(5), shape=(N,N)))
    K = L_init@L_init.T + jnp.eye(N)*1e-6
    obs_var = jnp.diag(K)
    noise_std = (jnp.diag(K).mean()**.5)/100.
    obs = obs_mean + random.normal(random.PRNGKey(55), shape=(N,1))*noise_std
    explicit_v_sample_estimate(obs, obs_mean, K, noise_std)

def test_general_high_noise() -> None:

    N = 500
    obs_mean = random.normal(random.PRNGKey(690), shape=(N,1))
    L_init = jnp.tril(random.normal(random.PRNGKey(1795), shape=(N,N)))
    K = L_init@L_init.T + jnp.eye(N)*1e-6
    obs_var = jnp.diag(K)
    noise_std = (jnp.diag(K).mean()**.5)/2.
    obs = obs_mean + random.normal(random.PRNGKey(1751), shape=(N,1))*noise_std
    explicit_v_sample_estimate(obs, obs_mean, K, noise_std)

