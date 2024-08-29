from __future__ import annotations

import dataclasses
from typing import Literal

from viser._messages import RemoveNotificationMessage, UpdateNotificationMessage
from viser.infra._infra import WebsockClientConnection


@dataclasses.dataclass
class _NotificationHandleState:
    websock_interface: WebsockClientConnection
    id: str
    title: str
    body: str
    loading: bool
    with_close_button: bool
    auto_close: int | Literal[False]


@dataclasses.dataclass
class NotificationHandle:
    """Handle for a notification in our visualizer."""

    _impl: _NotificationHandleState

    def _update_notification(self) -> None:
        m = UpdateNotificationMessage(
            self._impl.id,
            self._impl.title,
            self._impl.body,
            self._impl.loading,
            self._impl.with_close_button,
            self._impl.auto_close,
        )
        self._impl.websock_interface.queue_message(m)

    @property
    def title(self) -> str:
        """Title to display on the notification."""
        return self._impl.title

    @title.setter
    def title(self, title: str) -> None:
        if title == self._impl.title:
            return

        self._impl.title = title
        self._update_notification()

    @property
    def body(self) -> str:
        """Message to display on the notification body."""
        return self._impl.body

    @body.setter
    def body(self, body: str) -> None:
        if body == self._impl.body:
            return

        self._impl.body = body
        self._update_notification()

    @property
    def loading(self) -> bool:
        """Whether the notification shows loading icon."""
        return self._impl.loading

    @loading.setter
    def loading(self, loading: bool) -> None:
        if loading == self._impl.loading:
            return

        self._impl.loading = loading
        self._update_notification()

    @property
    def with_close_button(self) -> bool:
        """Whether the notification can be manually closed."""
        return self._impl.with_close_button

    @with_close_button.setter
    def with_close_button(self, with_close_button: bool) -> None:
        if with_close_button == self._impl.with_close_button:
            return

        self._impl.with_close_button = with_close_button
        self._update_notification()

    @property
    def auto_close(self) -> int | Literal[False]:
        """Time in ms before the notification automatically closes;
        otherwise False such that the notification never closes on its own."""
        return self._impl.auto_close

    @auto_close.setter
    def auto_close(self, auto_close: int | Literal[False]) -> None:
        if auto_close == self._impl.auto_close:
            return

        self._impl.auto_close = auto_close
        self._update_notification()

    def remove(self) -> None:
        self._impl.websock_interface.queue_message(
            RemoveNotificationMessage(self._impl.id)
        )
