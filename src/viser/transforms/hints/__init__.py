from typing import NamedTuple, Union

import numpy as onp

# Type aliases Numpy arrays; primarily for function inputs.

Array = onp.ndarray
"""Type alias for onp.ndarray."""

Scalar = Union[float, Array]
"""Type alias for `Union[float, Array]`."""


class RollPitchYaw(NamedTuple):
    """Tuple containing roll, pitch, and yaw Euler angles."""

    roll: Scalar
    pitch: Scalar
    yaw: Scalar


__all__ = [
    "Array",
    "Scalar",
    "RollPitchYaw",
]
