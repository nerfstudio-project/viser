from __future__ import annotations

import dataclasses

import numpy as onp
import numpy.typing as onpt
from typing_extensions import override

from . import _base, hints
from .utils import register_lie_group


@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)
@dataclasses.dataclass
class SO2(_base.SOBase):
    """Special orthogonal group for 2D rotations.

    Ported to numpy from `jaxlie.SO2`.

    Internal parameterization is `(cos, sin)`. Tangent parameterization is `(omega,)`.
    """

    # SO2-specific.

    unit_complex: onpt.NDArray[onp.floating]
    """Internal parameters. `(cos, sin)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = onp.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: hints.Scalar) -> SO2:
        """Construct a rotation object from a scalar angle."""
        cos = onp.cos(theta)
        sin = onp.sin(theta)
        return SO2(unit_complex=onp.array([cos, sin]))

    def as_radians(self) -> onpt.NDArray[onp.floating]:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()[..., 0]
        return radians

    # Factory.

    @classmethod
    @override
    def identity(cls) -> SO2:
        return SO2(unit_complex=onp.array([1.0, 0.0]))

    @classmethod
    @override
    def from_matrix(cls, matrix: onpt.NDArray[onp.floating]) -> SO2:
        assert matrix.shape == (2, 2)
        return SO2(unit_complex=onp.asarray(matrix[:, 0]))

    # Accessors.

    @override
    def as_matrix(self) -> onpt.NDArray[onp.floating]:
        cos_sin = self.unit_complex
        out = onp.array(
            [
                # [cos, -sin],
                cos_sin * onp.array([1, -1]),
                # [sin, cos],
                cos_sin[::-1],
            ]
        )
        assert out.shape == (2, 2)
        return out

    @override
    def parameters(self) -> onpt.NDArray[onp.floating]:
        return self.unit_complex

    # Operations.

    @override
    def apply(self, target: onpt.NDArray[onp.floating]) -> onpt.NDArray[onp.floating]:
        assert target.shape == (2,)
        return self.as_matrix() @ target  # type: ignore

    @override
    def multiply(self, other: SO2) -> SO2:
        return SO2(unit_complex=self.as_matrix() @ other.unit_complex)

    @classmethod
    @override
    def exp(cls, tangent: onpt.NDArray[onp.floating]) -> SO2:
        (theta,) = tangent
        cos = onp.cos(theta)
        sin = onp.sin(theta)
        return SO2(unit_complex=onp.array([cos, sin]))

    @override
    def log(self) -> onpt.NDArray[onp.floating]:
        return onp.arctan2(
            self.unit_complex[..., 1, None], self.unit_complex[..., 0, None]
        )

    @override
    def adjoint(self) -> onpt.NDArray[onp.floating]:
        return onp.eye(1)

    @override
    def inverse(self) -> SO2:
        return SO2(unit_complex=self.unit_complex * onp.array([1, -1]))

    @override
    def normalize(self) -> SO2:
        return SO2(unit_complex=self.unit_complex / onp.linalg.norm(self.unit_complex))

    #  @staticmethod
    #  @override
    #  def sample_uniform(key: hints.KeyArray) -> SO2:
    #      return SO2.from_radians(
    #          jax.random.uniform(key=key, minval=0.0, maxval=2.0 * onp.pi)
    #      )
