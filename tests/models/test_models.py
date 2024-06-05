"""
File: test_models.py
Description: Tests GPR and PSGP implementations
"""

import sys
sys.path.append('../../')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from jax import config, random, scipy, vmap
import jax.numpy as jnp
from jax.scipy import linalg
config.update("jax_enable_x64", True)

from tensorflow_probability.substrates import jax as tfp
triangle_transform = tfp.bijectors.FillScaleTriL(diag_shift=None)

import jax_pigp
from jax_pigp.models import GPR
from jax_pigp.utility_fns import softplus, add_noise


# slight numerical differences exist between the jax_pigp and sklearn results
SK_TOL = 1e-3

NUGGET = 1e-7
PSGP_TOL = 1e-6

def gen_data(data_opt):

    base_key = random.PRNGKey(data_opt['rng_seed'])

    s1, s2, s3 = random.split(base_key, 3)

    lb, ub = data_opt['domain']

    u_fn_vmap = vmap(data_opt['u_fn'])

    Xs = random.uniform(s1, minval=jnp.array(lb), maxval=jnp.array(ub), shape=(data_opt['Ntest'],len(ub)))

    us = u_fn_vmap(Xs)
    std_noise = jnp.std(us)*data_opt['noise_perc']
    obs_noise = random.normal(s2, shape=(data_opt['Ntrain'],1))*std_noise

    Xu = random.uniform(s3, minval=jnp.array(lb), maxval=jnp.array(ub), shape=(data_opt['Ntrain'],len(ub)))
    yu = u_fn_vmap(Xu).reshape(-1,1) + obs_noise

    return Xu, yu, Xs, std_noise


def sklearn_jax_comparison(data_opt):

    # generate simulation data
    Xu, yu, Xs, std_noise = gen_data(data_opt)

    ## sklearn code

    # initialise sklearn GP
    sklearn_kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    sklearn_gp     = GaussianProcessRegressor(kernel=sklearn_kernel, alpha=float(std_noise)**2, n_restarts_optimizer=9)

    # fit to training data
    sklearn_gp.fit(Xu, yu)

    # save trained hyper-parameter values (converted from log scale)
    sklearn_params = jnp.exp(sklearn_gp.kernel_.theta)

    # compute log ml value
    sklearn_obj = sklearn_gp.log_marginal_likelihood_value_

    # predict on test data
    sklearn_mean, sklearn_std = sklearn_gp.predict(Xs, return_std=True)

    ## jax_pigp code

    # initialise jax_pigp params
    params = {}
    params['amp'] = softplus(sklearn_params[0], inv=True)
    params['ls']  = softplus(sklearn_params[1], inv=True)
    params['noise_std'] = softplus(jnp.array([float(std_noise)]), inv=True)

    # intialise jax_pigp gp
    identity_trans = lambda f, a=1: f
    lin_ops = [identity_trans]
    mean_fn = lambda p,x: 0.
    kernel_fn = jax_pigp.kernels.squared_exponential
    heat_kernel_fns = jax_pigp.interdomain_moments.InterdomainKernelFns(kernel_fn, lin_ops)
    heat_mean_fns   = jax_pigp.interdomain_moments.InterdomainMeanFns(mean_fn,     lin_ops)
    gp_model = jax_pigp.models.GPR(heat_kernel_fns, heat_mean_fns)

    # store simulation data in training dataset
    train_data = jax_pigp.datasets.TrainDataLinearPDE([Xu], [yu])

    # compute log ml value
    jax_obj = gp_model.log_marginal_likelihood(params, train_data)

    # predict on test set
    jax_mean, jax_cov = gp_model.posterior_predict(params, train_data, Xs)
    jax_cov = jnp.diag(jax_cov)
    jax_mean = jax_mean[:,0]

    ## compare results
    assert jnp.allclose(jax_obj, sklearn_obj, rtol=SK_TOL)
    assert jnp.allclose(jax_mean, sklearn_mean, atol=SK_TOL)
    assert jnp.allclose(jax_cov**.5, sklearn_std, atol=SK_TOL)


def get_true_var_params(noise_var, Khh, Kyy, err):

    # prior covariance
    Lhh = linalg.cholesky(Khh, lower=True)

    Lyy           = linalg.cholesky(Kyy, lower=True)
    Lyyinv_Khh      = linalg.solve_triangular(Lyy, Khh, lower=True)
    Kyyinv_Khh = linalg.solve_triangular(Lyy.T, Lyyinv_Khh, lower=False)

    posterior_mean = jnp.matmul(Kyyinv_Khh.T, err)
    posterior_cov  = Khh - jnp.matmul(Khh, Kyyinv_Khh)
    A_true = linalg.cholesky(posterior_cov, lower=True)

    a_whitened_true = linalg.solve_triangular(Lhh, posterior_mean, lower=True)
    A_whitened_true = linalg.solve_triangular(Lhh, A_true, lower=True)

    return a_whitened_true, A_whitened_true

def psgp_test_fn(data_opt: dict) -> None:

    # generate simulation data
    X, y, Xs, std_noise = gen_data(data_opt)

    Ntrain = X.shape[0]//3
    Xu, yu = X[:Ntrain], y[:Ntrain]
    Xf, yf = X[Ntrain:], y[Ntrain:]

    D = Xu.shape[1]

    # initialise jax_pigp params
    params_init = {}
    params_init['amp'] = softplus(1., inv=True)
    params_init['ls']  = softplus(jnp.array([1.]*D), inv=True)
    params_init['noise_std'] = softplus(jnp.array([float(std_noise)]*3)*5., inv=True)

    identity_trans = lambda f, a=1: f
    lin_ops = [identity_trans]*2
    mean_fn = lambda p,x: (x[0]/3.).squeeze()
    kernel_fn = jax_pigp.kernels.squared_exponential
    kernel_fns = jax_pigp.interdomain_moments.InterdomainKernelFns(kernel_fn, lin_ops)
    mean_fns   = jax_pigp.interdomain_moments.InterdomainMeanFns(mean_fn,     lin_ops)

    # intialise jax_pigp gp
    exact_gp = jax_pigp.models.GPR(kernel_fns, mean_fns, nugget=NUGGET)

    # store simulation data in training dataset
    train_data = jax_pigp.datasets.TrainDataLinearPDE([Xu, Xf], [yu, yf])

    # compute log ml value
    exact_obj = exact_gp.log_marginal_likelihood(params_init, train_data)

    # predict on test set
    exact_mean, exact_cov = exact_gp.posterior_predict(params_init, train_data, Xs)

    # create psgp
    N_Lf = 1
    F = lambda p, d: d[0]
    train_data_approx = jax_pigp.datasets.TrainDataNonLinearPDE([Xu,Xf, Xf[:0]], [yu,yf, None], [Xu.shape[0], Xf.shape[0],0])
    approx_gp = jax_pigp.models.PSGP(kernel_fns, mean_fns, F, N_Lf, nugget=NUGGET)

    # compute true value of variational parameters given F is identity transformation
    noise_var = softplus(params_init['noise_std'])**2
    Khh = kernel_fns.get_Khh(params_init, train_data.Xtrain)
    Khh += jnp.eye(Khh.shape[0])*NUGGET
    Kyy = add_noise(Khh, noise_var, train_data.space_indicator)
    err = train_data.ytrain - mean_fns.get_mh(None, train_data.Xtrain)
    a_whitened_true, A_whitened_true = get_true_var_params(noise_var, Khh, Kyy, err)

    # write to parameter dict
    params_init['a_whitened']    =  a_whitened_true
    params_init['A_whitened_vec'] = triangle_transform.inverse(A_whitened_true)

    # predict on test set with psgp
    approx_mean, approx_cov = approx_gp.posterior_predict(params_init, train_data_approx, Xs)

    assert jnp.allclose(exact_mean, approx_mean, atol=PSGP_TOL)
    assert jnp.allclose(exact_cov, approx_cov, atol=PSGP_TOL)


def test_gpr1D() -> None:
    data_opt1D = {}
    data_opt1D['rng_seed'] = 0
    data_opt1D['Ntrain']   = 20
    data_opt1D['Ntest']    = 100
    data_opt1D['u_fn'] = lambda x: x*jnp.sin(x*jnp.pi)
    data_opt1D['u_fn'] = lambda x: x**2#jnp.sin(x*jnp.pi)
    data_opt1D['domain'] = [[0],[1]]
    data_opt1D['noise_perc'] = 0.1

    sklearn_jax_comparison(data_opt1D)

def test_gpr3D() -> None:
    data_opt3D = {}
    data_opt3D['rng_seed'] = 0
    data_opt3D['Ntrain']   = 200
    data_opt3D['Ntest']    = 250
    data_opt3D['u_fn'] = lambda x: x[0]**2 + x[1]*x[2] + x[1]
    data_opt3D['domain'] = [[0., -1., 1.],[1., 1., 2.]]
    data_opt3D['noise_perc'] = 0.55

    sklearn_jax_comparison(data_opt3D)


def test_psgp1D() -> None:

    data_opt1D = {}
    data_opt1D['rng_seed'] = 0
    data_opt1D['Ntrain']   = 101
    data_opt1D['Ntest']    = 202
    data_opt1D['u_fn'] = lambda x: x*jnp.sin(x*jnp.pi)
    data_opt1D['u_fn'] = lambda x: x**2#jnp.sin(x*jnp.pi)
    data_opt1D['domain'] = [[0],[1]]
    data_opt1D['noise_perc'] = 0.1

    psgp_test_fn(data_opt1D)

def test_psgp4D() -> None:

    data_opt4D = {}
    data_opt4D['rng_seed'] = 0
    data_opt4D['Ntrain']   = 201
    data_opt4D['Ntest']    = 250
    data_opt4D['u_fn'] = lambda x: x[0]**2 + x[1]*x[2] + x[1]*x[3] + x[0]
    data_opt4D['domain'] = [[0., -1., 1., 0.],[1., 1., 2., 2.]]
    data_opt4D['noise_perc'] = 0.25

    psgp_test_fn(data_opt4D)



