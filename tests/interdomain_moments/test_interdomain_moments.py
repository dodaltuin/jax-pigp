"""
File: test_interdomain_moments.py
Description: Tests mean / covariance values generated using the InterdomainMeanFns
and InterdomainKernelFns classes respectively
"""

# ensure that we can access jax_pigp
import sys
sys.path.append('../../')

from jax import config, random, scipy, vmap, jacrev
import jax.numpy as jnp
config.update("jax_enable_x64", True)

import jax_pigp
from jax_pigp.interdomain_moments import InterdomainMeanFns, InterdomainKernelFns
from jax_pigp.utility_fns import softplus
from jax_pigp.interdomain_moments.utils_vmap import vmap_kernel_fn

def apply_jac_sum(fn,argnum=1):
    r"""
    """

    # the Jacobian function
    Jfn = jacrev(fn, argnums=argnum)

    def jacobian_sum(params: dict, loc1, *loc2):

        Jvector = Jfn(params, loc1, *loc2)

        return Jvector.sum()

    return jacobian_sum


def identity(fn, argnum=1):
    return fn

def kfu_se_1D(params, x, z):
    r"""
    Evaluates squared-exponential kernel on pair of inputs loc1 and loc2 from the
    spatio-temporal domain, with kernel hyper-parameters stored in params
    """

    # extract kernel parameter values
    amp = softplus(params['amp'])
    ls  = softplus(params['ls'])[0]
    diff = (x-z).sum()

    # evaluate squared-exponential kernel at inputs loc1, loc2
    cov_val =  -(amp * diff * jnp.exp(-0.5 * (diff/ls)**2))/(ls**2)

    return cov_val.squeeze()

def kff_se_1D(params, x, z):
    r"""
    Evaluates squared-exponential kernel on pair of inputs loc1 and loc2 from the
    spatio-temporal domain, with kernel hyper-parameters stored in params
    """

    # extract kernel parameter values
    amp = softplus(params['amp'])
    ls  = softplus(params['ls'])[0]
    diff = (x-z).sum()

    mult_factor = -amp*(diff**2 - ls**2)/ls**4

    # evaluate squared-exponential kernel at inputs loc1, loc2
    cov_val =  mult_factor*jnp.exp(-0.5 * (diff/ls)**2)

    return cov_val.squeeze()

class KernelDataset1D:

    # squared exponential kernel
    kuu = jax_pigp.kernels.squared_exponential
    kuu_vmap = vmap_kernel_fn(kuu)
    kfu_vmap = vmap_kernel_fn(kfu_se_1D)
    kff_vmap = vmap_kernel_fn(kff_se_1D)

    # generate input data
    D = 1
    N0, N1, N2 = 1, 2, 5

    Xtrain = [random.normal(random.PRNGKey(2254), shape=(N0,D)),
              random.normal(random.PRNGKey(1112), shape=(N1,D))]
    Xtest = random.normal(random.PRNGKey(3344), shape=(N2,D))

    # list of linear operators
    lin_ops_testing = [identity, apply_jac_sum]

    # spatial indicators for each training input
    Xtrain_indices = [0, 1]

    params = {'amp': softplus(1., inv=True),
              'ls' : softplus(jnp.array([1.]*D), inv=True)}

    # covariance over training points
    # A variable scope error arises if the below is done
    # more compactly using a list comprehension
    K00 = kuu_vmap(params, Xtrain[0], Xtrain[0])
    K10 = kfu_vmap(params, Xtrain[1], Xtrain[0])
    K11 = kff_vmap(params, Xtrain[1], Xtrain[1])
    Khh_top = jnp.hstack([K00, K10.T])
    Khh_bot = jnp.hstack([K10, K11])
    Khh_manual = jnp.vstack([Khh_top,
                             Khh_bot])

    # cross covariance between training and test
    Kh0s = kuu_vmap(params, Xtrain[0], Xtest)
    Kh1s = kfu_vmap(params, Xtrain[1], Xtest)
    Khs_manual = jnp.vstack([Kh0s, Kh1s])

    # covariance over test points
    Kss_manual = kuu_vmap(params, Xtest, Xtest)

class KernelDataset4D:

    # squared exponential kernel
    kuu = jax_pigp.kernels.squared_exponential
    kuu_vmap = vmap_kernel_fn(kuu)

    # generate input data
    D = 4
    N0, N1, N2 = 41, 29, 154

    Xtrain = [random.normal(random.PRNGKey(54), shape=(N0,D)),
              random.normal(random.PRNGKey(12), shape=(N1,D))]
    Xtest = random.normal(random.PRNGKey(44), shape=(N2,D))

    # list of linear operators
    lin_ops_testing = [identity, identity, identity]

    # spatial indicators for each training input
    Xtrain_indices = [0, 0, 1]

    params = {'amp': softplus(1., inv=True),
              'ls' : softplus(jnp.array([1.]*D), inv=True)}

    # covariance over training points
    # A variable scope error arises if the below is done
    # more compactly using a list comprehension
    K00 = kuu_vmap(params, Xtrain[0], Xtrain[0])
    K01 = kuu_vmap(params, Xtrain[0], Xtrain[0])
    K02 = kuu_vmap(params, Xtrain[0], Xtrain[1])
    K11 = kuu_vmap(params, Xtrain[0], Xtrain[0])
    K12 = kuu_vmap(params, Xtrain[0], Xtrain[1])
    K22 = kuu_vmap(params, Xtrain[1], Xtrain[1])
    Khh_top = jnp.hstack([K00, K01, K02])
    Khh_mid = jnp.hstack([K01.T, K11, K12])
    Khh_bot = jnp.hstack([K02.T, K12.T, K22])
    Khh_manual = jnp.vstack([Khh_top,
                             Khh_mid,
                             Khh_bot])

    # cross covariance between training and test
    Kh0s = kuu_vmap(params, Xtrain[0], Xtest)
    Kh1s = kuu_vmap(params, Xtrain[0], Xtest)
    Kh2s = kuu_vmap(params, Xtrain[1], Xtest)
    Khs_manual = jnp.vstack([Kh0s, Kh1s, Kh2s])

    # covariance over test points
    Kss_manual = kuu_vmap(params, Xtest, Xtest)


class MeanDataset1D:


    # polynomial mean function
    mfn_testing  = lambda p, x: (x**2 - 2.*x - 1.).squeeze()
    dmfn_testing = lambda p, x: (2.*x - 2.).squeeze()

    # squared exponential kernel
    kuu_testing = jax_pigp.kernels.squared_exponential

    params = {}

    # generate input data
    D = 1
    N1, N2, N3 = 24, 53, 111
    #N1, N2, N3 = 2, 3, 5

    Xtrain = [random.normal(random.PRNGKey(71), shape=(N1,D)),
              random.normal(random.PRNGKey(19), shape=(N2,D))]
    Xtest = random.normal(random.PRNGKey(65), shape=(N3,D))

    # list of linear operators
    lin_ops_testing = [identity, apply_jac_sum, identity]

    # spatial indicators for each training input
    Xtrain_indices = [0, 1, 1]

    mh_manual = jnp.vstack([mfn_testing(None, Xtrain[0]).reshape(-1,1),
                            dmfn_testing(None, Xtrain[1]).reshape(-1,1),
                            mfn_testing(None, Xtrain[1]).reshape(-1,1)])
    ms_manual = mfn_testing(None, Xtest).reshape(-1,1)

class MeanDataset3D:


    # polynomial mean function in 3D
    def mfn_testing(p, loc):
        x, y, z = loc
        return x**3 - 2.*y**2 + 3*y + z -  y*z - 5.
    def dmfn_testing(p, loc):
        x, y, z = loc
        return 3.*x**2 - 4.*y + 3. - z + 1. - y

    mfn_testing_v  = vmap(mfn_testing, in_axes=[None,0])
    dmfn_testing_v = vmap(dmfn_testing, in_axes=[None,0])

    params = {}

    # generate input data
    D = 3
    N1, N2, N3 = 231, 112, 211
    N0, N1, N2 = 2, 3, 5

    Xtrain = [random.normal(random.PRNGKey(712), shape=(N0,D)),
              random.normal(random.PRNGKey(109), shape=(N1,D))]
    Xtest = random.normal(random.PRNGKey(650), shape=(N2,D))

    # list of linear operators
    lin_ops_testing = [identity, apply_jac_sum]

    # spatial indicators for each training input
    Xtrain_indices = [0, 1]

    mh_manual = jnp.vstack([mfn_testing_v(None, Xtrain[0]).reshape(-1,1),
                            dmfn_testing_v(None, Xtrain[1]).reshape(-1,1)])
    ms_manual = mfn_testing_v(None, Xtest).reshape(-1,1)


def kernel_testing_fn(data) -> None:

    testing_kernel_fns = InterdomainKernelFns(data.kuu, data.lin_ops_testing, data.Xtrain_indices)

    Khh_eval = testing_kernel_fns.get_Khh(data.params, data.Xtrain)
    Khs_eval = testing_kernel_fns.get_Khs(data.params, data.Xtrain, data.Xtest)
    Kss_eval = testing_kernel_fns.get_Kss(data.params, data.Xtest)

    assert jnp.allclose(Khh_eval, data.Khh_manual)
    assert jnp.allclose(Khs_eval, data.Khs_manual)
    assert jnp.allclose(Kss_eval, data.Kss_manual)


def mean_testing_fn(data) -> None:

    testing_mean_fns = InterdomainMeanFns(data.mfn_testing, data.lin_ops_testing, data.Xtrain_indices)

    mh_eval = testing_mean_fns.get_mh(data.params, data.Xtrain)
    ms_eval = testing_mean_fns.get_ms(data.params, data.Xtest)

    assert jnp.allclose(mh_eval, data.mh_manual)
    assert jnp.allclose(ms_eval, data.ms_manual)

def test_kernel_1D() -> None:
    kernel_testing_fn(KernelDataset1D)

def test_kernel_4D() -> None:
    kernel_testing_fn(KernelDataset4D)

def test_mean_1D() -> None:
    mean_testing_fn(MeanDataset1D)

def test_mean_3D() -> None:
    mean_testing_fn(MeanDataset3D)


