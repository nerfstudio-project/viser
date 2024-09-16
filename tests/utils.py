import functools
import random
from typing import Any, Callable, Tuple, Type, TypeVar, Union, cast

import numpy as onp
import numpy.typing as onpt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import viser.transforms as vtf

T = TypeVar("T", bound=vtf.MatrixLieGroup)


def sample_transform(
    Group: Type[T], batch_axes: Tuple[int, ...], dtype: onpt.DTypeLike
) -> T:
    """Sample a random transform from a group."""
    seed = random.getrandbits(32)
    strategy = random.randint(0, 2)

    if strategy == 0:
        # Uniform sampling.
        return cast(
            T,
            Group.sample_uniform(
                onp.random.default_rng(seed), batch_axes=batch_axes, dtype=dtype
            ),
        )
    elif strategy == 1:
        # Sample from normally-sampled tangent vector.
        return cast(
            T,
            Group.exp(
                onp.random.randn(*batch_axes, Group.tangent_dim).astype(dtype=dtype)
            ),
        )
    elif strategy == 2:
        # Sample near identity.
        return cast(
            T,
            Group.exp(
                onp.random.randn(*batch_axes, Group.tangent_dim).astype(dtype=dtype)
                * 1e-7
            ),
        )
    else:
        assert False


def general_group_test(
    f: Callable[[Type[vtf.MatrixLieGroup], Tuple[int, ...], onpt.DTypeLike], None],
    max_examples: int = 3,
) -> Callable[[Type[vtf.MatrixLieGroup], Tuple[int, ...], onpt.DTypeLike, Any], None]:
    """Decorator for defining tests that run on all group types."""

    # Disregard unused argument.
    def f_wrapped(
        Group: Type[vtf.MatrixLieGroup],
        batch_axes: Tuple[int, ...],
        dtype: onpt.DTypeLike,
        _random_module,
    ) -> None:
        f(Group, batch_axes, dtype)

    # Disable timing check (first run requires JIT tracing and will be slower).
    f_wrapped = settings(deadline=None, max_examples=max_examples)(f_wrapped)

    # Add _random_module parameter.
    f_wrapped = given(_random_module=st.random_module())(f_wrapped)

    # Parametrize tests with each group type.
    f_wrapped = pytest.mark.parametrize(
        "Group",
        [
            vtf.SO2,
            vtf.SE2,
            vtf.SO3,
            vtf.SE3,
        ],
    )(f_wrapped)

    # Parametrize tests with each group type.
    f_wrapped = pytest.mark.parametrize(
        "batch_axes",
        [
            (),
            (1,),
            (3, 1, 2, 1),
        ],
    )(f_wrapped)

    # Parametrize tests with each group type.
    f_wrapped = pytest.mark.parametrize(
        "dtype",
        [onp.float32, onp.float64],
    )(f_wrapped)
    return f_wrapped


general_group_test_faster = functools.partial(general_group_test, max_examples=5)


def assert_transforms_close(a: vtf.MatrixLieGroup, b: vtf.MatrixLieGroup):
    """Make sure two transforms are equivalent."""
    # Check matrix representation.
    assert_arrays_close(a.as_matrix(), b.as_matrix())

    # Flip signs for quaternions.
    # We use `jnp.asarray` here in case inputs are onp arrays and don't support `.at()`.
    p1 = a.parameters().copy()
    p2 = b.parameters().copy()
    if isinstance(a, vtf.SO3):
        p1 = p1 * onp.sign(onp.sum(p1, axis=-1, keepdims=True))
        p2 = p2 * onp.sign(onp.sum(p2, axis=-1, keepdims=True))
    elif isinstance(a, vtf.SE3):
        p1[..., :4] *= onp.sign(onp.sum(p1[..., :4], axis=-1, keepdims=True))
        p2[..., :4] *= onp.sign(onp.sum(p2[..., :4], axis=-1, keepdims=True))

    # Make sure parameters are equal.
    assert_arrays_close(p1, p2)


def assert_arrays_close(
    *arrays: Union[onpt.NDArray[onp.float64], float],
    rtol: float = 1e-3,
    atol: float = 1e-4,
):
    """Make sure two arrays are close. (and not NaN)"""
    for array1, array2 in zip(arrays[:-1], arrays[1:]):
        assert onp.asarray(array1).dtype == onp.asarray(array2).dtype
        onp.testing.assert_allclose(array1, array2, rtol=rtol, atol=atol)
        assert not onp.any(onp.isnan(array1))
        assert not onp.any(onp.isnan(array2))
