from __future__ import annotations

import dataclasses
from typing import Literal

from viser._messages import RemoveNotificationMessage, UpdateNotificationMessage
from viser.infra._infra import WebsockClientConnection


@dataclasses.dataclass
class NotificationHandle:
    """Handle for a notification in our visualizer."""

    _websock_interface: WebsockClientConnection
    _id: str
    _title: str
    _body: str
    _loading: bool = False
    _with_close_button: bool = True
    _auto_close: int | Literal[False] = False

    def _update_notification(self) -> None:
        m = UpdateNotificationMessage(
            self._id,
            self._title,
            self._body,
            self._loading,
            self._with_close_button,
            self._auto_close,
        )
        self._websock_interface.queue_message(m)

    @property
    def title(self) -> str:
        """Title to display on the notification."""
        return self._title

    @title.setter
    def title(self, title: str) -> None:
        if title == self._title:
            return

        self._title = title
        self._update_notification()

    @property
    def body(self) -> str:
        """Message to display on the notification body."""
        return self._body

    @body.setter
    def body(self, body: str) -> None:
        if body == self._body:
            return

        self._body = body
        self._update_notification()

    @property
    def loading(self) -> bool:
        """Whether the notification shows loading icon."""
        return self._loading

    @loading.setter
    def loading(self, loading: bool) -> None:
        if loading == self._loading:
            return

        self._loading = loading
        self._update_notification()

    @property
    def with_close_button(self) -> bool:
        """Whether the notification can be manually closed."""
        return self._with_close_button

    @with_close_button.setter
    def with_close_button(self, with_close_button: bool) -> None:
        if with_close_button == self._with_close_button:
            return

        self._with_close_button = with_close_button
        self._update_notification()

    @property
    def auto_close(self) -> int | Literal[False]:
        """Time in ms before the notification automatically closes;
        otherwise False such that the notification never closes on its own."""
        return self._auto_close

    @auto_close.setter
    def auto_close(self, auto_close: int | Literal[False]) -> None:
        if auto_close == self._auto_close:
            return

        self._auto_close = auto_close
        self._update_notification()

    def remove(self) -> None:
        self._websock_interface.queue_message(RemoveNotificationMessage(self._id))
