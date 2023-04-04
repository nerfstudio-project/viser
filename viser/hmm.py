import dataclasses
from typing import Generic, LiteralString, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class Container(Generic[T]):
    """Dummy generic class."""
    inner: T


TLiteralString = TypeVar("TLiteralString", bound=LiteralString)


def wrap_with_default(
    x: TLiteralString | None,
    default: TLiteralString,
) -> Container[TLiteralString]:
    """Wrap an input with a dummy container object."""
    if x is None:
        x = default
    reveal_type(Container(x))
    return out
