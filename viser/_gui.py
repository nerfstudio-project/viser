from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, TypeVar

if TYPE_CHECKING:
    from ._message_api import MessageApi


T = TypeVar("T")


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    name: str
    source: MessageApi
    value: T
    last_updated: float
    update_cb: List[Callable[[T], None]]
    cleanup_cb: Optional[Callable[[], Any]] = None


@dataclasses.dataclass(frozen=True)
class GuiHandle(Generic[T]):
    """Handle for a particular GUI input in our visualizer."""

    # Let's shove private implementation details in here...
    _impl: _GuiHandleState[T]

    def on_update(self, func: Callable[[T], None]) -> Callable[[T], None]:
        """Attach a function to call whenever this field updates."""
        self._impl.update_cb.append(func)
        return func

    def value(self) -> T:
        return self._impl.value

    def last_updated(self) -> float:
        return self._impl.last_updated

    # TODO
    #
    # def set_value(
    #     self, value: T, client_id: Union[ClientId, Literal["all"]] = "all"
    # ) -> T:
    #     """Set the value of this field."""
    #     raise NotImplementedError()
    #
    # def detach(self) -> None:
    #     """Remove this GUI element from the visualizer."""
    #     raise NotImplementedError()
