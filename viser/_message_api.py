from __future__ import annotations

import abc
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from typing_extensions import LiteralString, ParamSpec

from . import _messages
from ._gui import GuiHandle, _GuiHandleState

if TYPE_CHECKING:
    from ._server import ClientId


P = ParamSpec("P")


# TODO(by): I had to drop a Concatenate[] to make this signature work. Seems like probably a
# pyright bug?
#
# https://github.com/microsoft/pyright/issues/4813
def _wrap_message(
    message_cls: Callable[P, _messages.Message]
) -> Callable[[Callable], Callable[P, None]]:
    """Wrap a message type."""

    def inner(self: MessageApi, *args: P.args, **kwargs: P.kwargs) -> None:
        message = message_cls(*args, **kwargs)
        self._queue(message)

    return lambda _: inner  # type: ignore


IntOrFloat = TypeVar("IntOrFloat", int, float)
TLiteral = TypeVar("TLiteral", bound=LiteralString)


class MessageApi(abc.ABC):
    """Interface for all commands we can use to send messages over a websocket connection."""

    def __init__(self) -> None:
        self._handle_from_gui_name: Dict[str, _GuiHandleState[Any]] = {}
        self._incoming_handlers: List[Callable[[_messages.Message], None]] = []
        self._incoming_handlers.append(lambda msg: _handle_gui_updates(self, msg))

    def add_gui_checkbox(self, name: str, initial_value: bool) -> GuiHandle[bool]:
        """Add a checkbox to the GUI."""
        assert isinstance(initial_value, bool)
        return _add_gui_impl(
            self, name, initial_value, leva_conf={"value": initial_value}
        )

    def add_gui_select(self, name: str, options: List[TLiteral]) -> GuiHandle[TLiteral]:
        """Add a dropdown to the GUI."""
        assert len(options) > 0
        initial_value = options[0]
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={"value": initial_value, "options": options},
        )

    def add_gui_slider(
        self,
        name: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: Optional[IntOrFloat],
        initial_value: IntOrFloat,
    ) -> GuiHandle[IntOrFloat]:
        """Add a dropdown to the GUI."""
        assert max > min
        if step is not None:
            assert step < (max - min)
        assert max > initial_value > min

        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={"value": initial_value, "min": min, "max": max, "step": step},
        )

    @_wrap_message(_messages.CameraFrustumMessage)
    def add_camera_frustum(self):
        ...

    @_wrap_message(_messages.FrameMessage)
    def add_frame(self):
        ...

    @_wrap_message(_messages.PointCloudMessage)
    def add_point_cloud(self):
        ...

    @_wrap_message(_messages.ImageMessage.encode)
    def add_image(self):
        ...

    @_wrap_message(_messages.RemoveSceneNodeMessage)
    def remove_scene_node(self):
        ...

    @_wrap_message(_messages.BackgroundImageMessage.encode)
    def set_background_image(self):
        ...

    @_wrap_message(_messages.ResetSceneMessage)
    def reset_scene(self):
        ...

    def _handle_incoming_message(
        self, client_id: ClientId, message: _messages.Message
    ) -> None:
        for cb in self._incoming_handlers:
            cb(message)

    @abc.abstractmethod
    def _queue(self, message: _messages.Message) -> None:
        """Abstract method for sending messages."""
        ...


def _handle_gui_updates(self: MessageApi, message: _messages.Message) -> None:
    if not isinstance(message, _messages.GuiUpdateMessage):
        return

    if message.name not in self._handle_from_gui_name:
        return

    self._handle_from_gui_name[message.name].value = message.value
    self._handle_from_gui_name[message.name].last_updated = time.time()

    for cb in self._handle_from_gui_name[message.name].update_cb:
        cb(message.value)


def _add_gui_impl(
    api: MessageApi, name: str, initial_value: Any, leva_conf: dict
) -> GuiHandle[Any]:
    """Private helper for adding a simple GUI element."""

    handle = _GuiHandleState(
        name,
        source=api,
        value=initial_value,
        last_updated=time.time(),
        update_cb=[],
    )
    api._handle_from_gui_name[name] = handle
    handle.cleanup_cb = lambda: api._handle_from_gui_name.pop(name)

    api._queue(
        _messages.AddGuiInputMessage(
            name=name,
            leva_conf=leva_conf,
        )
    )
    return GuiHandle(handle)
