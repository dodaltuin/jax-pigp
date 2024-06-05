"""
File: test_kl_divergence.py
Description: Tests gaussian_kl() function
"""

# ensure that we can access jax_pigp
import sys
sys.path.append('../')

from jax import config, random, scipy
import jax.numpy as jnp
config.update("jax_enable_x64", True)

from jax_pigp.kl_divergence import gaussian_kl


def test_oned() -> None:
    """
    Check gaussian_kl against hand calculation in 1D
    """
    m1 = jnp.array([[2.]])
    L1 = jnp.array([[1.]])
    m2 = jnp.array([[0.5]])
    K2 = jnp.array([[2.25]])

    kl_val = gaussian_kl(m1, L1, m2, K2)

    # hand calculated value
    kl_val_calc = 0.6276873303303866

    assert jnp.allclose(kl_val, kl_val_calc)


def test_twod() -> None:
    """
    Check gaussian_kl against hand calculation in 2D
    """
    m1 = jnp.array([[0.], [1.]])
    L1 = jnp.array([[1., 0.],
                    [0.5, 2.]])
    m2 = jnp.array([[1.5], [0.5]])
    K2 = jnp.array([[3., 1.2],
                    [1.2, 2.1]])

    kl_val = gaussian_kl(m1, L1, m2, K2)

    # hand calculated value
    kl_val_calc = 1.25014982

    assert jnp.allclose(kl_val, kl_val_calc)


def test_zero() -> None:
    """
    Check zero divergence between identical N-dimensional
    Gaussians
    """
    N = 50
    m = random.normal(random.PRNGKey(0), shape=(N,1))
    L_init = jnp.tril(random.normal(random.PRNGKey(5), shape=(N,N)))
    K = L_init@L_init.T + jnp.eye(N)*1e-6
    L = scipy.linalg.cholesky(K, lower=True)

    kl_val = gaussian_kl(m, L, m, K)

    assert jnp.allclose(0., kl_val)
