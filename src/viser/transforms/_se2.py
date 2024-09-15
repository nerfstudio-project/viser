from __future__ import annotations

import dataclasses
from typing import Tuple, cast

import numpy as onp
import numpy.typing as onpt
from typing_extensions import override

from . import _base, hints
from ._so2 import SO2
from .utils import broadcast_leading_axes, get_epsilon


@dataclasses.dataclass(frozen=True)
class SE2(
    _base.SEBase[SO2],
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
):
    """Special Euclidean group for proper rigid transforms in 2D. Broadcasting
    rules are the same as for numpy.

    Ported to numpy from `jaxlie.SE2`.

    Internal parameterization is `(cos, sin, x, y)`. Tangent parameterization is `(vx,
    vy, omega)`.
    """

    # SE2-specific.

    unit_complex_xy: onpt.NDArray[onp.floating]
    """Internal parameters. `(cos, sin, x, y)`. Shape should be `(*, 4)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = onp.round(self.unit_complex_xy[..., :2], 5)
        xy = onp.round(self.unit_complex_xy[..., 2:], 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: hints.Scalar, y: hints.Scalar, theta: hints.Scalar) -> SE2:
        """Construct a transformation from standard 2D pose parameters.

        This is not the same as integrating over a length-3 twist.
        """
        cos = onp.cos(theta)
        sin = onp.sin(theta)
        return SE2(unit_complex_xy=onp.stack([cos, sin, x, y], axis=-1))

    # SE-specific.

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO2,
        translation: onpt.NDArray[onp.floating],
    ) -> SE2:
        assert translation.shape[-1:] == (2,)
        rotation, translation = broadcast_leading_axes((rotation, translation))
        return SE2(
            unit_complex_xy=onp.concatenate(
                [rotation.unit_complex, translation], axis=-1
            )
        )

    @override
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2])

    @override
    def translation(self) -> onpt.NDArray[onp.floating]:
        return self.unit_complex_xy[..., 2:]

    # Factory.

    @classmethod
    @override
    def identity(
        cls, batch_axes: Tuple[int, ...] = (), dtype: onpt.DTypeLike = onp.float64
    ) -> SE2:
        return SE2(
            unit_complex_xy=onp.broadcast_to(
                onp.array([1.0, 0.0, 0.0, 0.0], dtype=dtype), (*batch_axes, 4)
            )
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: onpt.NDArray[onp.floating]) -> SE2:
        assert matrix.shape[-2:] == (3, 3) or matrix.shape[-2:] == (2, 3)
        # Currently assumes bottom row is [0, 0, 1].
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[..., :2, :2]),
            translation=matrix[..., :2, 2],
        )

    # Accessors.

    @override
    def parameters(self) -> onpt.NDArray[onp.floating]:
        return self.unit_complex_xy

    @override
    def as_matrix(self) -> onpt.NDArray[onp.floating]:
        cos, sin, x, y = onp.moveaxis(self.unit_complex_xy, -1, 0)
        out = onp.stack(
            [
                cos,
                -sin,
                x,
                sin,
                cos,
                y,
                onp.zeros_like(x),
                onp.zeros_like(x),
                onp.ones_like(x),
            ],
            axis=-1,
        ).reshape((*self.get_batch_axes(), 3, 3))
        return out

    # Operations.

    @classmethod
    @override
    def exp(cls, tangent: onpt.NDArray[onp.floating]) -> SE2:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L558
        # Also see:
        # > http://ethaneade.com/lie.pdf

        assert tangent.shape[-1:] == (3,)

        theta = tangent[..., 2]
        use_taylor = onp.abs(theta) < get_epsilon(tangent.dtype)

        # Shim to avoid NaNs in onp.where branches, which cause failures for
        # reverse-mode AD in JAX. This isn't needed for vanilla numpy.
        safe_theta = cast(
            onp.ndarray,
            onp.where(
                use_taylor,
                onp.ones_like(theta),  # Any non-zero value should do here.
                theta,
            ),
        )

        theta_sq = theta**2
        sin_over_theta = cast(
            onp.ndarray,
            onp.where(
                use_taylor,
                1.0 - theta_sq / 6.0,
                onp.sin(safe_theta) / safe_theta,
            ),
        )
        one_minus_cos_over_theta = cast(
            onp.ndarray,
            onp.where(
                use_taylor,
                0.5 * theta - theta * theta_sq / 24.0,
                (1.0 - onp.cos(safe_theta)) / safe_theta,
            ),
        )

        V = onp.stack(
            [
                sin_over_theta,
                -one_minus_cos_over_theta,
                one_minus_cos_over_theta,
                sin_over_theta,
            ],
            axis=-1,
        ).reshape((*tangent.shape[:-1], 2, 2))
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_radians(theta),
            translation=onp.einsum("...ij,...j->...i", V, tangent[..., :2]),
        )

    @override
    def log(self) -> onpt.NDArray[onp.floating]:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L160
        # Also see:
        # > http://ethaneade.com/lie.pdf

        theta = self.rotation().log()[..., 0]

        cos = onp.cos(theta)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        use_taylor = onp.abs(cos_minus_one) < get_epsilon(theta.dtype)

        # Shim to avoid NaNs in onp.where branches, which cause failures for
        # reverse-mode AD in JAX. This isn't needed for vanilla numpy.
        safe_cos_minus_one = onp.where(
            use_taylor,
            onp.ones_like(cos_minus_one),  # Any non-zero value should do here.
            cos_minus_one,
        )

        half_theta_over_tan_half_theta = onp.where(
            use_taylor,
            # Taylor approximation.
            1.0 - theta**2 / 12.0,
            # Default.
            -(half_theta * onp.sin(theta)) / safe_cos_minus_one,
        )

        V_inv = onp.stack(
            [
                half_theta_over_tan_half_theta,
                half_theta,
                -half_theta,
                half_theta_over_tan_half_theta,
            ],
            axis=-1,
        ).reshape((*theta.shape, 2, 2))

        tangent = onp.concatenate(
            [
                onp.einsum("...ij,...j->...i", V_inv, self.translation()),
                theta[..., None],
            ],
            axis=-1,
        )
        return tangent

    @override
    def adjoint(self: SE2) -> onpt.NDArray[onp.floating]:
        cos, sin, x, y = onp.moveaxis(self.unit_complex_xy, -1, 0)
        return onp.stack(
            [
                cos,
                -sin,
                y,
                sin,
                cos,
                -x,
                onp.zeros_like(x),
                onp.zeros_like(x),
                onp.ones_like(x),
            ],
            axis=-1,
        ).reshape((*self.get_batch_axes(), 3, 3))

    @classmethod
    @override
    def sample_uniform(
        cls,
        rng: onp.random.Generator,
        batch_axes: Tuple[int, ...] = (),
        dtype: onpt.DTypeLike = onp.float64,
    ) -> SE2:
        return SE2.from_rotation_and_translation(
            SO2.sample_uniform(rng, batch_axes=batch_axes, dtype=type),
            rng.uniform(low=-1.0, high=1.0, size=(*batch_axes, 2)).astype(dtype),
        )
