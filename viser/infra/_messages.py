"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import abc
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, cast

import msgpack
import numpy as onp

if TYPE_CHECKING:
    from ._core import ClientId
else:
    ClientId = Any


def _prepare_for_serialization(value: Any) -> Any:
    """Prepare any special types for serialization. Currently just maps numpy arrays to
    their underlying data buffers."""

    if isinstance(value, onp.ndarray):
        return value.data if value.data.c_contiguous else value.copy().data
    else:
        return value


T = TypeVar("T", bound="Message")


class Message(abc.ABC):
    """Base message type for controlling our viewer."""

    excluded_self_client: Optional[ClientId] = None
    """Don't send this message to a particular client. Useful when a client wants to
    send synchronization information to other clients."""

    def serialize(self) -> bytes:
        """Convert a Python Message object into bytes."""
        mapping = {k: _prepare_for_serialization(v) for k, v in vars(self).items()}
        out = msgpack.packb({"type": type(self).__name__, **mapping})
        assert isinstance(out, bytes)
        return out

    @staticmethod
    def deserialize(message: bytes) -> Message:
        """Convert bytes into a Python Message object."""
        mapping = msgpack.unpackb(message)

        # msgpack deserializes to lists by default, but all of our annotations use
        # tuples.
        mapping = {
            k: tuple(v) if isinstance(v, list) else v for k, v in mapping.items()
        }
        message_type = Message._subclass_from_type_string()[
            cast(str, mapping.pop("type"))
        ]
        return message_type(**mapping)

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

        For example: if we send 1000 "set value" messages for the same gui element, we
        should only keep the latest message.
        """
