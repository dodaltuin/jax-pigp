"""
File: psgp.py
Description: Implements physics and symmetry informed Gaussian process (PSGP) regression in JAX
"""

import jax.numpy as jnp
from jax import vmap
from jax.scipy import linalg
from functools import partial

from tensorflow_probability.substrates import jax as tfp

triangle_transform = tfp.bijectors.FillScaleTriL(diag_shift=None)

from ..utility_fns import softplus
from ..datasets import TrainDataNonLinearPDE
from ..base import Array, MeanAndCovariance, ArrayList
from ..interdomain_moments import InterdomainMeanFns, InterdomainKernelFns
from ..expectations import compute_exp_log_prob_vmap, estimate_log_prob_yf
from ..kl_divergence import gaussian_kl

from typing import Callable


class PSGP:
    r"""
    Physics and Symmetry informed Gaussian Processess (PSGPs).

    Implements PSGP assuming Gaussian observation noise in each function space.

    Training is performed by optimising the elbo w.r.t. the
    mean/kernel/noise/PDE/variational parameters.

    Prediction at test points of interest can be made using
    posterior_predict.

    Latent vectors in different function spaces seperated by a linear
    operator accomodated through the use of an instance of
    InterdomainMeanFns to compute mean terms, and an instance of
    InterdomainKernelFns to compute covariance terms.
    """

    def __init__(
        self,
        kernel_fns: InterdomainKernelFns,
        mean_fns: InterdomainMeanFns,
        F: Callable,  # algebraic equation representing the PDE
        len_d: int,  # number of individual linear operators used in specification of F
        n_samples: int = 20000,  # number of samples to use when estimating log_pyf
        noise_trans: Callable = softplus,
        nugget: float = 1e-7,
    ):

        self.kernel_fns = kernel_fns
        self.mean_fns = mean_fns
        self.noise_trans = noise_trans
        self.nugget = nugget

        F_vmap = vmap(F, in_axes=[None, 0])

        # function which uses Monte-Carlo sampling to estimate the expected log prob. of f space data
        self.estimate_log_pyf = partial(
            estimate_log_prob_yf, F_vmap=F_vmap, len_d=len_d, n_samples=n_samples
        )

        # function which explitly calculates the expected log prob. of u and g space observation data
        self.expected_log_prob = compute_exp_log_prob_vmap

    def get_common_values(
        self, params: dict, train_data: TrainDataNonLinearPDE, compute_elbo: bool = True
    ):
        r"""
        Computes matrices and arrays needed for both training and prediction

        Args:
            params: dict containing values of mean/kernel/noise/PDE parameters
            train_data: training data from possibly multiple function spaces
            compute_elbo: equal to True if elbo() is being evaluated and False if posterior_predict() is

        Returns
        -------
            ArrayList: matrices/vectors required for elbo() and posterior_predict() methods
        """

        # whitened version of variational mean parameter a
        a_whitened = params["a_whitened"]

        # whitened version of lower triangular variational covariance parameter A
        A_whitened = triangle_transform.forward(params["A_whitened_vec"])

        # identity matrix of dimension (Nh x Nh), where Nh is the dim of the latent space
        In = jnp.eye(a_whitened.shape[0], dtype=jnp.float64)

        # prior covariance matrix over latent vector h, with nugget term on diagonal for stability
        Khh = self.kernel_fns.get_Khh(params, train_data.Xtrain) + In * self.nugget
        Lhh = linalg.cholesky(Khh, lower=True)

        # prior mean vector over latent vector h
        mh = self.mean_fns.get_mh(params, train_data.Xtrain) if compute_elbo else 0.0
        ## Note: if evaluating posterior_predict() (i.e. if compute_elbo is False), then we do not need
        ## to compute mh as this gets subtracted out in the final prediction formula for mu_s

        # un-whitened version of mean parameter a
        a = Lhh @ a_whitened + mh

        # un-whitened version of covariance parameter A
        A = Lhh @ A_whitened

        # approximate/variational posterior covariance matrix over h
        S = A @ A.T

        if compute_elbo:
            return In, Khh, Lhh, a_whitened, a, A_whitened, A, S
        else:
            return Lhh, a, A, S

    def split_variational_parameters(
        self, a: Array, S: Array, Nu: int, Nf: int, Ng: int
    ) -> ArrayList:
        r"""
        Splits variational mean parameter a and covariance parameter S1 according
        to which function space (i.e. u/f or g space) each element belongs to

        ## Note: We assume u-space input points (Xu) come before f-space inputs (Xf), which
        ## in turn comes before g-space input points (Xg) in train_data.Xtrain

        Args:
            a: variational mean vector
            S: variational covariance matrix
            Nu/Nf/Ng: Number of observations in u/f/g space. We assume Nf > 0

        Returns
        -------
            ArrayList: elements of a/S1 split according to u/f/g space
        """

        # first index corresponding to g-space data. Note we can have Ng=0
        g_start_index = a.shape[0] - Ng

        # each element of a corresponds to a data point in u, g or f-space
        a_u, a_f, a_g = jnp.split(a, [Nu, g_start_index])

        # we only need the individual variance values for parameter relating
        # to u/g space data because the terms relating to yu/yg in the ELBO
        # can be evaluated in closed form
        var_u, _, var_g = jnp.split(jnp.diag(S), [Nu, g_start_index])

        # for parameters corresponding to f-space data we need the full
        # covariance matrix to form a Monte-Carlo sample estimate of the
        # term relating to yf in the ELBO
        S_ff = S[Nu:g_start_index, Nu:g_start_index]

        return a_u, a_f, a_g, var_u, S_ff, var_g

    def elbo(self, params: dict, train_data: TrainDataNonLinearPDE) -> float:
        r"""
        Computes evidence lower bound (ELBO) on the log marginal likelihood of the
        training data
        """

        # number of observations in each function space, i.e. u, f or g-space
        Nu, Nf, Ng = train_data.Nu, train_data.Nf, train_data.Ng

        # extract noise std levels in each function space
        sigma_u, sigma_f, sigma_g = self.noise_trans(params["noise_std"])

        # matrices / vectors required for computation of the terms in the ELBO
        In, Khh, Lhh, a_whitened, a, A_whitened, A, S = self.get_common_values(
            params, train_data, compute_elbo=True
        )

        # split the variational parameters depending on which function space they
        # correspond do
        a_u, a_f, a_g, var_u, S_ff, var_g = self.split_variational_parameters(
            a, S, Nu, Nf, Ng
        )

        # KL divergence between whitened variational posterior and isotropic Gaussian
        kl = gaussian_kl(a_whitened, A_whitened, a_whitened * 0.0, In)

        # expectated log probability of u-space observations yu
        exp_log_pyu = (
            self.expected_log_prob(train_data.yu, a_u, var_u, sigma_u).sum()
            if Nu > 0
            else 0.0
        )

        # expectated log probability of f-space (i.e. PDE) observations yf, evaluated by sampling
        exp_log_pyf = self.estimate_log_pyf(
            params, a_f, S_ff, train_data.yf, sigma_f.squeeze() ** 2
        )
        ## Note: we assume that Nf > 0 - if not, the nonlinearPIGP is not required

        # expectated log probability of g-space (i.e. ISC) observations yg
        exp_log_pyg = (
            self.expected_log_prob(train_data.yg, a_g, var_g, sigma_g).sum()
            if Ng > 0
            else 0.0
        )

        # ELBO is comprised of the four above terms
        elbo = -kl + exp_log_pyu + exp_log_pyf + exp_log_pyg

        return elbo

    def posterior_predict(
        self, params: dict, train_data: TrainDataNonLinearPDE, Xs: Array
    ) -> MeanAndCovariance:
        r"""
        Computes the mean and covariance matrix of the approximate Gaussian posterior
        predictive distribution at test inputs Xs, i.e.

            \hat{p}(us | Xs, train_data, params)

        Returns
        -------
            MeanAndCovariance: mean vector (mu_s) and covariance matrix (Sigma_s)
            of approximate posterior distribution \hat{p} over test outputs us.

        """

        Lhh, a, A, S = self.get_common_values(params, train_data, compute_elbo=False)

        # prior mean vector / covariance matrix over test outputs us
        ms = self.mean_fns.get_ms(params, Xs)
        Kss = self.kernel_fns.get_Kss(params, Xs)

        # cross covariance matrix between h and us
        Khs = self.kernel_fns.get_Khs(params, train_data.Xtrain, Xs)

        # weight matrix W from formula for p(us|h)
        W_T = linalg.solve_triangular(
            Lhh.T, linalg.solve_triangular(Lhh, Khs, lower=True)
        )
        W = W_T.T

        # approximate/variational posterior mean vector over us
        mu_s = ms + W @ a

        # approximate/variational posterior covariance matrix over us
        Sigma_s = Kss - W @ (Khs - S @ W_T)

        return mu_s, Sigma_s
