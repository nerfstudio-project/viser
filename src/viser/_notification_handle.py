from __future__ import annotations

import dataclasses
from typing import Any, Literal

from typing_extensions import override

from viser._assignable_props_api import AssignablePropsBase

from ._messages import NotificationMessage, NotificationProps, RemoveNotificationMessage
from .infra._infra import WebsockClientConnection


@dataclasses.dataclass
class _NotificationHandleState:
    websock_interface: WebsockClientConnection
    uuid: str
    props: NotificationProps


class NotificationHandle(
    NotificationProps, AssignablePropsBase[_NotificationHandleState]
):
    """Handle for a notification in our visualizer."""

    def __init__(self, impl: _NotificationHandleState) -> None:
        self._impl = impl

    @override
    def _queue_update(self, name: str, value: Any) -> None:
        """Queue an update message with the property change."""
        # For notifications, we'll just send the whole props object when a
        # property is reassigned. Deduplication in the message buffer will
        # debounce this when multiple properties are updated in succession.
        del name, value
        self._sync_with_client("update")

    def _sync_with_client(self, mode: Literal["show", "update"]) -> None:
        msg = NotificationMessage(mode, self._impl.uuid, self._impl.props)
        self._impl.websock_interface.queue_message(msg)

    def remove(self) -> None:
        self._impl.websock_interface.get_message_buffer().remove_from_buffer(
            # Don't send outdated GUI updates to new clients.
            # This is brittle...
            lambda message: getattr(message, "uuid", None) == self._impl.uuid
        )
        msg = RemoveNotificationMessage(self._impl.uuid)
        self._impl.websock_interface.queue_message(msg)
