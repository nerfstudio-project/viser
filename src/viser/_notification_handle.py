from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

from ._messages import NotificationMessage, NotificationProps, RemoveNotificationMessage
from .infra._infra import WebsockClientConnection


@dataclasses.dataclass
class _NotificationHandleState:
    websock_interface: WebsockClientConnection
    id: str
    props: NotificationProps


class NotificationHandle(NotificationProps):
    """Handle for a notification in our visualizer."""

    def __init__(self, impl: _NotificationHandleState) -> None:
        self._impl = impl

    # Support property-style read/write. Similar to `_OverridableScenePropApi`.
    if not TYPE_CHECKING:

        def __setattr__(self, name: str, value: Any) -> None:
            if name in NotificationProps.__annotations__:
                setattr(self._impl.props, name, value)
                self._sync_with_client("update")
            else:
                return object.__setattr__(self, name, value)

        def __getattr__(self, name: str) -> Any:
            if name in NotificationProps.__annotations__:
                return getattr(self._impl.props, name)
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    def _sync_with_client(self, mode: Literal["show", "update"]) -> None:
        msg = NotificationMessage(mode, self._impl.id, self._impl.props)
        self._impl.websock_interface.queue_message(msg)

    def remove(self) -> None:
        msg = RemoveNotificationMessage(self._impl.id)
        self._impl.websock_interface.queue_message(msg)
