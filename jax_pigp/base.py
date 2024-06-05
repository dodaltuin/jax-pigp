"""
File: base.py
Description: functionality for code typing
"""

import jax.numpy as jnp

from typing import Tuple, Sequence, Callable

FloatValue = float

Array = jnp.ndarray

IntegerList = Sequence[int]

CallableList = Sequence[Callable]

MeanAndCovariance = Tuple[Array, Array]

ArrayList = Sequence[Array]
