"""
File: datasets.py
Description: functionality for storing observational and PDE data
"""

import jax.numpy as jnp

from dataclasses import dataclass

from .base import Array, ArrayList, IntegerList


class TrainDataLinearPDE:
    r"""
    Holds data for problems involving linear PDEs

    For linear PDEs, the observations in different function spaces are jointly
    Gaussian, allowing us to concatenate all output together into a single
    array (self.ytrain).
    """

    def __init__(self, Xtrain: ArrayList, ytrain: ArrayList):

        # training inputs in each space are held in a list
        self.Xtrain = Xtrain

        # for linear PDEs we can stack observations from each function space together
        self.ytrain = jnp.vstack(ytrain)

        # indicates which function space each element of self.ytrain belongs to
        space_indicator_list = [jnp.zeros(X.shape[0]) + i for i, X in enumerate(Xtrain)]

        # convert to a jax array
        self.space_indicator = jnp.concatenate(space_indicator_list).astype(jnp.int32)


class TrainDataNonLinearPDE:
    r"""
    Holds data for problems involving nonlinear PDEs

    For nonlinear PDEs, u/f/g space data must be handled seperately in the
    (whitened) ELBO. For this reason, seperate attributes self.yu, self.yf and
    self.yg are defined
    """

    def __init__(self, Xtrain: ArrayList, ytrain: ArrayList, N_obs: IntegerList):

        # number of observations in each space - Nu/Ng = 0 supported
        self.Nu, self.Nf, self.Ng = N_obs

        # observations in different function spaces are handled seperately
        self.yu, self.yf, self.yg = ytrain
        ## Note: if there is no u-space data, set yu=None in ytrain above, and
        ## similarly if there is no g-space data

        # training inputs in each space are held in a list
        self.Xtrain = Xtrain


@dataclass
class TestData:
    Xs: Array  # input locations of test points
    us: Array  # test output values at Xs
