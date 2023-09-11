from __future__ import annotations

import dataclasses
from typing import cast

import numpy as onp
import numpy.typing as onpt
from typing_extensions import override

from . import _base
from ._so3 import SO3
from .utils import get_epsilon, register_lie_group


def _skew(omega: onpt.NDArray[onp.floating]) -> onpt.NDArray[onp.floating]:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = omega
    return onp.array(
        [  # type: ignore
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


@register_lie_group(
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
)
@dataclasses.dataclass
class SE3(_base.SEBase[SO3]):
    """Special Euclidean group for proper rigid transforms in 3D.

    Ported to numpy from `jaxlie.SE3`.

    Internal parameterization is `(qw, qx, qy, qz, x, y, z)`. Tangent parameterization
    is `(vx, vy, vz, omega_x, omega_y, omega_z)`.
    """

    # SE3-specific.

    wxyz_xyz: onpt.NDArray[onp.floating]
    """Internal parameters. wxyz quaternion followed by xyz translation."""

    @override
    def __repr__(self) -> str:
        quat = onp.round(self.wxyz_xyz[..., :4], 5)
        trans = onp.round(self.wxyz_xyz[..., 4:], 5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    # SE-specific.

    @staticmethod
    @override
    def from_rotation_and_translation(
        rotation: SO3,
        translation: onpt.NDArray[onp.floating],
    ) -> SE3:
        assert translation.shape == (3,)
        return SE3(wxyz_xyz=onp.concatenate([rotation.wxyz, translation]))

    @override
    @classmethod
    def from_translation(cls, translation: onpt.NDArray[onp.floating]) -> "SE3":
        return SE3.from_rotation_and_translation(SO3.identity(), translation)

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> onpt.NDArray[onp.floating]:
        return self.wxyz_xyz[..., 4:]

    # Factory.

    @staticmethod
    @override
    def identity() -> SE3:
        return SE3(wxyz_xyz=onp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @override
    def from_matrix(matrix: onpt.NDArray[onp.floating]) -> SE3:
        assert matrix.shape == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    # Accessors.

    @override
    def as_matrix(self) -> onpt.NDArray[onp.floating]:
        out = onp.eye(4)
        out[:3, :3] = self.rotation().as_matrix()
        out[:3, 3] = self.translation()
        return out
        #  return (
        #      onp.eye(4)
        #      .at[:3, :3]
        #      .set(self.rotation().as_matrix())
        #      .at[:3, 3]
        #      .set(self.translation())
        #  )

    @override
    def parameters(self) -> onpt.NDArray[onp.floating]:
        return self.wxyz_xyz

    # Operations.

    @staticmethod
    @override
    def exp(tangent: onpt.NDArray[onp.floating]) -> SE3:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L761

        # (x, y, z, omega_x, omega_y, omega_z)
        assert tangent.shape == (6,)

        rotation = SO3.exp(tangent[3:])

        theta_squared = tangent[3:] @ tangent[3:]
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        # Shim to avoid NaNs in onp.where branches, which cause failures for
        # reverse-mode AD. (note: this is needed in JAX, but not in numpy)
        theta_squared_safe = cast(
            onpt.NDArray[onp.floating],
            onp.where(
                use_taylor,
                1.0,  # Any non-zero value should do here.
                theta_squared,
            ),
        )
        del theta_squared
        theta_safe = onp.sqrt(theta_squared_safe)

        skew_omega = _skew(tangent[3:])
        V = onp.where(
            use_taylor,
            rotation.as_matrix(),
            (
                onp.eye(3)
                + (1.0 - onp.cos(theta_safe)) / (theta_squared_safe) * skew_omega
                + (theta_safe - onp.sin(theta_safe))
                / (theta_squared_safe * theta_safe)
                * (skew_omega @ skew_omega)
            ),
        )

        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=V @ tangent[:3],
        )

    @override
    def log(self) -> onpt.NDArray[onp.floating]:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()
        theta_squared = omega @ omega
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        skew_omega = _skew(omega)

        # Shim to avoid NaNs in onp.where branches, which cause failures for
        # reverse-mode AD. (note: this is needed in JAX, but not in numpy)
        theta_squared_safe = onp.where(
            use_taylor,
            1.0,  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = onp.sqrt(theta_squared_safe)
        half_theta_safe = theta_safe / 2.0

        V_inv = onp.where(
            use_taylor,
            onp.eye(3) - 0.5 * skew_omega + (skew_omega @ skew_omega) / 12.0,
            (
                onp.eye(3)
                - 0.5 * skew_omega
                + (
                    1.0
                    - theta_safe
                    * onp.cos(half_theta_safe)
                    / (2.0 * onp.sin(half_theta_safe))
                )
                / theta_squared_safe
                * (skew_omega @ skew_omega)
            ),
        )
        return onp.concatenate([V_inv @ self.translation(), omega])

    @override
    def adjoint(self) -> onpt.NDArray[onp.floating]:
        R = self.rotation().as_matrix()
        return onp.block(
            [
                [R, _skew(self.translation()) @ R],
                [onp.zeros((3, 3)), R],
            ]
        )

    # @staticmethod
    # @override
    # def sample_uniform(key: hints.KeyArray) -> SE3:
    #     key0, key1 = jax.random.split(key)
    #     return SE3.from_rotation_and_translation(
    #         rotation=SO3.sample_uniform(key0),
    #         translation=jax.random.uniform(
    #             key=key1, shape=(3,), minval=-1.0, maxval=1.0
    #         ),
    #     )
