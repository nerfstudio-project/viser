import abc
from typing import ClassVar, Generic, Type, TypeVar, Union, overload

import numpy as onp
import numpy.typing as onpt
from typing_extensions import Self, final, override

from . import hints

GroupType = TypeVar("GroupType", bound="MatrixLieGroup")
SEGroupType = TypeVar("SEGroupType", bound="SEBase")


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties.
    # > These will be set in `_utils.register_lie_group()`.

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(self, parameters: onpt.NDArray[onp.floating], /):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    # Shared implementations.

    @overload
    def __matmul__(self, other: hints.Array) -> onpt.NDArray[onp.floating]:
        ...

    @overload
    def __matmul__(self: GroupType, other: GroupType) -> GroupType:
        ...

    def __matmul__(
        self: GroupType, other: Union[GroupType, hints.Array]
    ) -> Union[GroupType, onpt.NDArray[onp.floating]]:
        """Overload for the `@` operator.

        Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`.
        """
        if isinstance(other, onp.ndarray):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)
        else:
            assert False, f"Invalid argument type for `@` operator: {type(other)}"

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(cls: Type[GroupType]) -> GroupType:
        """Returns identity element.

        Returns:
            Identity element.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls: Type[GroupType], matrix: hints.Array) -> GroupType:
        """Get group member from matrix representation.

        Args:
            matrix: Matrix representaiton.

        Returns:
            Group member.
        """

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> onpt.NDArray[onp.floating]:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> onpt.NDArray[onp.floating]:
        """Get underlying representation."""

    # Operations.

    @abc.abstractmethod
    def apply(self, target: hints.Array) -> onpt.NDArray[onp.floating]:
        """Applies group action to a point.

        Args:
            target: Point to transform.

        Returns:
            Transformed point.
        """

    @abc.abstractmethod
    def multiply(self: Self, other: Self) -> Self:
        """Composes this transformation with another.

        Returns:
            self @ other
        """

    @classmethod
    @abc.abstractmethod
    def exp(cls: Type[GroupType], tangent: hints.Array) -> GroupType:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent: Tangent vector to take the exponential of.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def log(self) -> onpt.NDArray[onp.floating]:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self) -> onpt.NDArray[onp.floating]:
        """Computes the adjoint, which transforms tangent vectors between tangent
        spaces.

        More precisely, for a transform `GroupType`:
        ```
        GroupType @ exp(omega) = exp(Adj_T @ omega) @ GroupType
        ```

        In robotics, typically used for transforming twists, wrenches, and Jacobians
        across different reference frames.

        Returns:
            Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inverse(self: Self) -> Self:
        """Computes the inverse of our transform.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def normalize(self: Self) -> Self:
        """Normalize/projects values and returns.

        Returns:
            GroupType: Normalized group member.
        """

    #  @classmethod
    #  @abc.abstractmethod
    #  def sample_uniform(cls: Type[GroupType], key: hints.KeyArray) -> GroupType:
    #      """Draw a uniform sample from the group. Translations (if applicable) are in the
    #      range [-1, 1].
    #
    #      Args:
    #          key: PRNG key, as returned by `jax.random.PRNGKey()`.
    #
    #      Returns:
    #          Sampled group member.
    #      """


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)


class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups.

    Each SE(N) group member contains an SO(N) rotation, as well as an N-dimensional
    translation vector.
    """

    # SE-specific interface.

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(
        cls: Type[SEGroupType],
        rotation: ContainedSOType,
        translation: hints.Array,
    ) -> SEGroupType:
        """Construct a rigid transform from a rotation and a translation.

        Args:
            rotation: Rotation term.
            translation: translation term.

        Returns:
            Constructed transformation.
        """

    @final
    @classmethod
    def from_rotation(cls: Type[SEGroupType], rotation: ContainedSOType) -> SEGroupType:
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=onp.zeros(cls.space_dim, dtype=rotation.parameters().dtype),
        )

    @classmethod
    @abc.abstractmethod
    def from_translation(
        cls: Type[SEGroupType], translation: onpt.NDArray[onp.floating]
    ) -> SEGroupType:
        """Construct a transform from a translation term."""

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> onpt.NDArray[onp.floating]:
        """Returns a transform's translation term."""

    # Overrides.

    @final
    @override
    def apply(self, target: hints.Array) -> onpt.NDArray[onp.floating]:
        return self.rotation() @ target + self.translation()  # type: ignore

    @override
    @final
    def multiply(self: SEGroupType, other: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    @final
    @override
    def inverse(self: SEGroupType) -> SEGroupType:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    @final
    @override
    def normalize(self: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )
