from __future__ import annotations

import abc
from functools import cached_property
from typing import Any, Dict, Generic, Protocol, TypeVar, get_type_hints

import numpy as np
import numpy.typing as npt

# Type variable for props


class HasProps(Protocol):
    props: Any  # One of the `*Props` objects in _messages.py.


TImpl = TypeVar("TImpl", bound=HasProps)


def colors_to_uint8(colors: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert intensity values to uint8. We assume the range [0,1] for floats, and
    [0,255] for integers. Accepts any shape."""
    if colors.dtype != np.uint8:
        if np.issubdtype(colors.dtype, np.floating):
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        if np.issubdtype(colors.dtype, np.integer):
            colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


class OverridablePropsBase(Generic[TImpl]):
    """Base class for all API objects with overridable properties."""

    _impl: TImpl

    def _cast_array_dtypes(
        self, prop_hints: Dict[str, Any], prop_name: str, value: np.ndarray
    ) -> np.ndarray:
        """Helper to cast array values to the correct data type."""
        hint = prop_hints[prop_name]
        if hint == npt.NDArray[np.float32]:
            return value.astype(np.float32)
        elif hint == npt.NDArray[np.float16]:
            return value.astype(np.float16)
        if hint == npt.NDArray[np.uint8] and "color" in prop_name:
            # ^TODO: revisit name heuristic here...
            value = colors_to_uint8(value)
        return value

    @cached_property
    def _prop_hints(self) -> Dict[str, Any]:
        return get_type_hints(type(self._impl.props))

    @abc.abstractmethod
    def _queue_update(self, name: str, value: Any) -> None:
        """Queue an update message with the property change."""


def props_setattr(self, name: str, value: Any) -> None:
    if name == "_impl":
        return object.__setattr__(self, name, value)

    # If it's a property with a setter, use the setter.
    prop = getattr(self.__class__, name, None)
    if isinstance(prop, property) and prop.fset is not None:
        prop.fset(self, value)
        return

    # Try to handle as a props field.
    if name in self._prop_hints:
        # Handle array type casting.
        if isinstance(value, np.ndarray):
            value = self._cast_array_dtypes(self._prop_hints, name, value)

        current_value = getattr(self._impl.props, name)

        # Skip update if value hasn't changed.
        if isinstance(current_value, np.ndarray):
            if np.array_equal(current_value, value):
                return
        elif current_value == value:
            return

        # Update the value based on type.
        if isinstance(value, np.ndarray):
            if hasattr(current_value, "dtype"):
                # Ensure consistent dtype.
                if value.dtype != current_value.dtype:
                    value = value.astype(current_value.dtype)

            # In-place update for same shape arrays.
            if hasattr(current_value, "shape") and value.shape == current_value.shape:
                current_value[:] = value
            else:
                setattr(self._impl.props, name, value.copy())
        else:
            # Non-array properties
            setattr(self._impl.props, name, value)

        self._queue_update(name, value)
    else:
        return object.__setattr__(self, name, value)


def props_getattr(self, name: str) -> Any:
    if name in self._prop_hints:
        return getattr(self._impl.props, name)
    else:
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


OverridablePropsBase.__setattr__ = props_setattr  # type: ignore
OverridablePropsBase.__getattr__ = props_getattr  # type: ignore
