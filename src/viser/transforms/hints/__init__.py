from typing import Union

import numpy as onp
import numpy.typing as onpt

# Type aliases Numpy arrays; primarily for function inputs.

Scalar = Union[float, onpt.NDArray[onp.floating]]
"""Type alias for `Union[float, Array]`."""


__all__ = [
    "Scalar",
]
