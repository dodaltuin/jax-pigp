"""
File: interdomain_moments.py
Description: functionality to evaluate mean and kernel functions where data is
available in different function spaces, each of which is found by applying a linear
operator to the solution space. Note that for data from solution space itself, this
operator is simply the identity transformation.
"""

from jax.numpy import hstack, vstack

from .utils_vmap import vmap_mean_fn, vmap_kernel_fn, check_shapes
from ..base import Array, ArrayList, IntegerList, Callable, CallableList


class InterdomainMeanFns:
    r"""
    Initialises and evaluates mean functions according to observations in
    different function spaces, each of which is found by applying a linear
    operator to solution space (u-space)
    """

    def __init__(
        self,
        mean_fn: Callable,
        linear_operators: CallableList,
        Xtrain_indices: IntegerList = None,
    ):
        r"""
        Args:
            mean_fn: base mean function in u space (i.e. solution space)
            linear_operators: list, each element of which contains a linear
            differential operator, the forms of which will depend on the PDE
            Xtrain_indices: list, the ith element of which gives the index in
            train_data.Xtrain that the ith element of linear_operators
            corresponds to

        NOTE:
            > We assume the first argument to mean_fn is a dictionary of parameter
            values
            > Xtrain_indices does not need to be initialised for linear PDEs
        """

        # check that len(linear_operators) == len(Xtrain_indices)
        check_shapes(linear_operators, Xtrain_indices)

        self.m_u = mean_fn
        # vmap for application to multiple inputs simultaneously
        self.m_u_vmap = vmap_mean_fn(self.m_u)

        self.linear_operators = linear_operators
        self.n_l = len(linear_operators)

        # apply self.linear_operators to self.m_u
        self.get_interdomain_mean_fns()

        self.space_indices = (
            [i for i in range(self.n_l)] if Xtrain_indices is None else Xtrain_indices
        )

    def get_mh(self, params: dict, Xtrain: ArrayList) -> Array:
        r"""
        Compute prior mean vector over latent vector h at Xtrain, given
        parameter values specified in params
        """
        # unpack spatial indicator vector
        l = self.space_indices
        # evaluate mean function from each individual space at the locations in
        # input space it corresponds to, then stack all results together
        mh = vstack(
            [
                self.mean_fns_vmap[i](params, Xtrain[l[i]]).reshape(-1, 1)
                for i in range(self.n_l)
            ]
        )
        return mh

    def get_ms(self, params: dict, Xs: Array) -> Array:
        r"""
        Compute prior mean vector over latent function values us at test inputs
        Xs, given parameter values specified in params
        """
        # reshape to ensure that ms has shape (Ns,1)
        ms = self.m_u_vmap(params, Xs).reshape(-1, 1)
        return ms

    def get_interdomain_mean_fns(self) -> None:
        r"""
        Define the mean functions in each different space by applying the
        linear operators in self.linear_operators to the base mean function
        self.m_u
        """
        self.mean_fns = [op(self.m_u) for op in self.linear_operators]

        # vmap for application to multiple inputs simultaneously
        self.mean_fns_vmap = [vmap_mean_fn(m) for m in self.mean_fns]


class InterdomainKernelFns:
    r"""
    Initialises and evaluates kernel functions according to observations in
    different function spaces, each of which is found by applying a linear
    operator to solution space (u-space)
    """

    def __init__(
        self,
        kernel_fn: Callable,
        linear_operators: CallableList,
        Xtrain_indices: IntegerList = None,
    ):
        r"""
        Args:
            kernel_fn: base kernel function in u space
            linear_operators: list, each element of which contains a linear
            differential operator, the forms of which will depend on the PDE
            Xtrain_indices: list, the ith element of which gives the index in
            train_data.Xtrain that the ith element of linear_operators
            corresponds to

        NOTE:
            > We assume the first argument to kernel_fn is a dictionary of parameter
            values
            > Xtrain_indices does not need to be initialised for linear PDEs
        """

        # check that len(linear_operators) == len(Xtrain_indices)
        check_shapes(linear_operators, Xtrain_indices)

        self.kuu = kernel_fn
        # vmap for application to multiple inputs simultaneously
        self.kuu_vmap = vmap_kernel_fn(self.kuu)

        self.linear_operators = linear_operators
        self.n_l = len(linear_operators)

        # apply self.linear_operators to self.kuu
        self.get_interdomain_kernel_fns()

        self.space_indices = (
            [i for i in range(self.n_l)] if Xtrain_indices is None else Xtrain_indices
        )

    def get_Khh(self, params: dict, Xtrain: ArrayList) -> Array:
        r"""
        Compute prior covariance matrix over latent vector h at Xtrain, given
        parameter values specified in params
        """
        # unpack some attributes
        l = self.space_indices
        nl = self.n_l
        kfns = self.kernel_fns_vmap
        # evaluate kernel and cross kernel functions to each function-space
        # pair at the appropriate location in input space, then stack all
        # results together
        Khh = hstack(
            [
                vstack(
                    [kfns[i][j](params, Xtrain[l[i]], Xtrain[l[j]]) for i in range(nl)]
                )
                for j in range(nl)
            ]
        )
        return Khh

    def get_Khs(self, params: dict, Xtrain: ArrayList, Xs: Array) -> Array:
        r"""
        Compute prior cross-covariance matrix between latent vector h at Xtrain
        and latent function values us at Xs, given parameter values specified
        in params
        """
        # unpack some attributes
        nl = self.n_l
        cross_kfns = self.cross_kernels_vmap
        l = self.space_indices
        # evaluate cross covariance functions from each individual space at
        # appropriate locations in Xtrain, against test points in Xs
        Khs = vstack([cross_kfns[i](params, Xtrain[l[i]], Xs) for i in range(nl)])
        return Khs

    def get_Kss(self, params: dict, Xs: Array) -> Array:
        r"""
        Compute prior covariance matrix over latent function values us at test
        inputs Xs, given parameter values specified in params
        """
        Kss = self.kuu_vmap(params, Xs, Xs)
        return Kss

    def get_interdomain_kernel_fns(self) -> None:
        r"""
        Define the kernel functions in each different space and cross kernels
        between spaces by applying the linear operators in self.linear_operators
        to the base kernel function self.kuu
        """
        # cross kernels gives covariance between operators in different spaces
        self.cross_kernels = [op(self.kuu, 1) for op in self.linear_operators]

        # re-apply differential operators to other input argument to get kernel
        # functions corresponding to each space
        self.kernel_fns = [
            [op(k, 2) for op in self.linear_operators] for k in self.cross_kernels
        ]

        # vmap for application to multiple inputs simultaneously
        self.kernel_fns_vmap = [
            [vmap_kernel_fn(k) for k in row] for row in self.kernel_fns
        ]
        self.cross_kernels_vmap = [vmap_kernel_fn(k) for k in self.cross_kernels]
