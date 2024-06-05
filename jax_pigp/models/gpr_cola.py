"""
File: gpr.py
Description: Implements Gaussian process regression in JAX
"""

import jax.numpy as jnp
from jax.scipy import linalg
import cola
from cola.ops import Dense
from cola.linalg import Cholesky

from ..utility_fns import softplus, add_noise
from ..datasets import TrainDataLinearPDE
from ..base import MeanAndCovariance, ArrayList, Array, FloatValue
from ..interdomain_moments import InterdomainMeanFns, InterdomainKernelFns

from typing import Callable

ALG = Cholesky()


class GPRcola:
    r"""
    Gaussian Process Regression (GPR).

    Implements GPR assuming Gaussian observation noise.

    Training is performed by optimising the log_marginal_likihood
    of the training data w.r.t. mean/kernel/noise/PDE parameters.

    Prediction at test points of interest can be made using
    posterior_predict.

    Observations in multiple function spaces seperated by a linear
    operator accomodated through the use of an instance of
    InterdomainMeanFns to compute mean terms, and an instance of
    InterdomainKernelFns to compute covariance terms.
    """

    def __init__(
        self,
        kernel_fns: InterdomainKernelFns,
        mean_fns: InterdomainMeanFns,
        noise_trans: Callable = softplus,
        nugget: float = 1e-8,
    ):
        self.kernel_fns = kernel_fns
        self.mean_fns = mean_fns
        self.noise_trans = noise_trans
        self.nugget = nugget

    def get_common_values(
        self, params: dict, train_data: TrainDataLinearPDE
    ) -> ArrayList:
        r"""
        Computes matrices and arrays needed for both training and prediction

        Args:
            params: dict containing values of mean/kernel/noise/PDE parameters
            train_data: training data from possibly multiple function spaces

        Returns
        -------
            ArrayList: vector err, lower triangular matrix Lyy and vector Lyyinv_err
        """

        # prediction error of prior mean function on training data
        err = Dense(train_data.ytrain - self.mean_fns.get_mh(params, train_data.Xtrain))

        # extract noise variances from each function space of interest simultaneously
        noise_var = self.noise_trans(params["noise_std"]) ** 2 + self.nugget

        # prior covariance matrix over latent vector of function values (h)
        Khh = Dense(self.kernel_fns.get_Khh(params, train_data.Xtrain))

        # add noise on diagonal to get prior covariance matrix over observational data
        Kyy = cola.PSD(add_noise(Khh, noise_var, train_data.space_indicator))
        Kyy_inv = cola.inv(Kyy, alg=ALG)

        Kyy_inv_err = Kyy_inv @ err

        return err, Kyy, Kyy_inv_err

    def log_marginal_likelihood(
        self, params: dict, train_data: TrainDataLinearPDE
    ) -> FloatValue:
        r"""
        Computes the log marginal likelihood

        Returns
        -------
            float: log marginal likelihood of training observations given params, i.e.

                \log p(train_data.ytrain | params).
        """
        err, Kyy, Kyy_inv_err = self.get_common_values(params, train_data)

        lml = -0.5 * (
            (err.T @ Kyy_inv_err)[0, 0]
            + cola.logdet(Kyy, ALG)
            + jnp.log(2 * jnp.pi) * err.shape[0]
        )

        return lml  # [0,0]

        return -(
            jnp.sum(jnp.log(jnp.diag(Lyy)))
            + 0.5 * (Lyyinv_err**2).sum()
            + 0.5 * (err.shape[0]) * jnp.log(2 * jnp.pi)
        )

    def posterior_predict(
        self, params: dict, train_data: TrainDataLinearPDE, Xs: Array
    ) -> MeanAndCovariance:
        r"""
        Computes the posterior predictive distribution at test inputs Xs

        Returns
        -------
            MeanAndCovariance: mean vector (mu) and covariance matrix (Sigma)
            of posterior distribution over test outputs us. i.e.

                p(us | Xs, train_data.ytrain, params).
        """

        err, Lyy, Lyyinv_err = self.get_common_values(params, train_data)
        Kyyinv_err = linalg.solve_triangular(Lyy.T, Lyyinv_err, lower=False)

        # prior mean and covariance at test inputs Xs
        ms = self.mean_fns.get_ms(params, Xs)
        Kss = self.kernel_fns.get_Kss(params, Xs)

        # covariance between training and test points
        Khs = self.kernel_fns.get_Khs(params, train_data.Xtrain, Xs)

        # posterior mean over function values at test inputs
        mu = ms + Khs.T @ Kyyinv_err

        # posterior covariance over function values at test inputs
        Lyyinv_Khs = linalg.solve_triangular(Lyy, Khs, lower=True)
        Sigma = Kss - Lyyinv_Khs.T @ Lyyinv_Khs

        return mu, Sigma
