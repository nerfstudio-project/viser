"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import abc
import functools
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, cast

import msgpack
import numpy as onp
from typing_extensions import get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from ._infra import ClientId
else:
    ClientId = Any


def _prepare_for_serialization(value: Any, annotation: Type) -> Any:
    """Prepare any special types for serialization."""

    # Coerce some scalar types: if we've annotated as float / int but we get an
    # onp.float32 / onp.int64, for example, we should cast automatically.
    if annotation is float:
        return float(value)
    if annotation is int:
        return int(value)

    # Recursively handle tuples.
    if get_origin(annotation) is tuple:
        if isinstance(value, onp.ndarray):
            assert False, (
                "Expected a tuple, but got an array... missing a cast somewhere?"
                f" {value}"
            )

        out = []
        args = get_args(annotation)
        if len(args) >= 1:
            if len(args) >= 2 and args[1] == ...:
                args = (args[0],) * len(value)
            elif len(value) != len(args):
                warnings.warn(f"[viser] {value} does not match annotation {annotation}")

            for i, v in enumerate(value):
                out.append(
                    # Hack to be OK with wrong type annotations.
                    # https://github.com/nerfstudio-project/nerfstudio/pull/1805
                    _prepare_for_serialization(v, args[i]) if i < len(args) else v
                )
            return tuple(out)

    # For arrays, we serialize underlying data directly. The client is responsible for
    # reading using the correct dtype.
    if isinstance(value, onp.ndarray):
        return value.data if value.data.c_contiguous else value.copy().data

    return value


T = TypeVar("T", bound="Message")


@functools.lru_cache(maxsize=None)
def get_type_hints_cached(cls: Type[Any]) -> Dict[str, Any]:
    return get_type_hints(cls)  # type: ignore


class Message(abc.ABC):
    """Base message type for server/client communication."""

    excluded_self_client: Optional[ClientId] = None
    """Don't send this message to a particular client. Useful when a client wants to
    send synchronization information to other clients."""

    def as_serializable_dict(self) -> Dict[str, Any]:
        """Convert a Python Message object into bytes."""
        hints = get_type_hints_cached(type(self))
        out = {
            k: _prepare_for_serialization(v, hints[k]) for k, v in vars(self).items()
        }
        out["type"] = type(self).__name__
        return out

    @classmethod
    def deserialize(cls, message: bytes) -> Message:
        """Convert bytes into a Python Message object."""
        mapping = msgpack.unpackb(message)

        # msgpack deserializes to lists by default, but all of our annotations use
        # tuples.
        mapping = {
            k: tuple(v) if isinstance(v, list) else v for k, v in mapping.items()
        }
        message_type = cls._subclass_from_type_string()[cast(str, mapping.pop("type"))]

        # If annotated as a float but we got an integer, cast to float. These
        # are both `number` in Javascript.
        def coerce_floats(value: Any, annotation: Type[Any]) -> Any:
            if annotation is float:
                return float(value)
            elif get_origin(annotation) is tuple:
                return tuple(
                    coerce_floats(value[i], typ)
                    for i, typ in enumerate(get_args(annotation))
                )
            else:
                return value

        type_hints = get_type_hints(message_type)
        mapping = {k: coerce_floats(v, type_hints[k]) for k, v in mapping.items()}
        return message_type(**mapping)  # type: ignore

    @classmethod
    @functools.lru_cache(maxsize=100)
    def _subclass_from_type_string(cls: Type[T]) -> Dict[str, Type[T]]:
        subclasses = cls.get_subclasses()
        return {s.__name__: s for s in subclasses}

    @classmethod
    def get_subclasses(cls: Type[T]) -> List[Type[T]]:
        """Recursively get message subclasses."""

        def _get_subclasses(typ: Type[T]) -> List[Type[T]]:
            out = []
            for sub in typ.__subclasses__():
                out.append(sub)
                out.extend(_get_subclasses(sub))
            return out

        return _get_subclasses(cls)

    @abc.abstractmethod
    def redundancy_key(self) -> str:
        """Returns a unique key for this message, used for detecting redundant
        messages.

        For example: if we send 1000 "set value" messages for the same GUI element, we
        should only keep the latest message.
        """
