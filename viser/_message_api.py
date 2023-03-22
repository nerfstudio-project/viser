from __future__ import annotations

import abc
import contextlib
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

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
    """Interface for all commands we can use to send messages over a websocket connection.

    Should be implemented by both our global server object (for broadcasting) and by
    invidividual clients."""

    def __init__(self) -> None:
        self._handle_state_from_gui_name: Dict[str, _GuiHandleState[Any]] = {}
        self._incoming_handlers: List[
            Callable[[ClientId, _messages.Message], None]
        ] = []
        self._incoming_handlers.append(
            lambda client_id, msg: _handle_gui_updates(self, client_id, msg)
        )
        self._gui_folder_label = "User"

    @contextlib.contextmanager
    def gui_folder(self, label: str) -> Generator[None, None, None]:
        """Context for placing all GUI elements into a particular folder.

        We currently only support one folder level."""
        old_folder_label = self._gui_folder_label
        self._gui_folder_label = label
        yield
        self._gui_folder_label = old_folder_label

    def add_gui_button(self, name: str, disabled: bool = False) -> GuiHandle[bool]:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`."""
        return _add_gui_impl(
            self,
            name,
            initial_value=False,
            leva_conf={"type": "BUTTON", "settings": {"disabled": disabled}},
            is_button=True,
        )

    def add_gui_checkbox(
        self, name: str, initial_value: bool, disabled: bool = False
    ) -> GuiHandle[bool]:
        """Add a checkbox to the GUI."""
        assert isinstance(initial_value, bool)
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={"value": initial_value, "disabled": disabled},
        )

    def add_gui_text(
        self, name: str, initial_value: str, disabled: bool = False
    ) -> GuiHandle[str]:
        """Add a text input to the GUI."""
        assert isinstance(initial_value, str)
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={"value": initial_value, "disabled": disabled},
        )

    def add_gui_number(
        self, name: str, initial_value: IntOrFloat, disabled: bool = False
    ) -> GuiHandle[IntOrFloat]:
        """Add a number input to the GUI."""
        assert isinstance(initial_value, (int, float))
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={"value": initial_value, "disabled": disabled},
        )

    def add_gui_vector2(
        self,
        name: str,
        initial_value: Tuple[float, float],
        step: Optional[float] = None,
        disabled: bool = False,
    ) -> GuiHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI."""
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={"value": initial_value, "step": step, "disabled": disabled},
        )

    def add_gui_vector3(
        self,
        name: str,
        initial_value: Tuple[float, float, float],
        step: Optional[float] = None,
        lock: bool = False,
        disabled: bool = False,
    ) -> GuiHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI."""
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={
                "value": initial_value,
                "step": step,
                "lock": lock,
                "disabled": disabled,
            },
        )

    def add_gui_select(
        self,
        name: str,
        options: List[TLiteral],
        initial_value: Optional[TLiteral] = None,
        disabled: bool = False,
    ) -> GuiHandle[TLiteral]:
        """Add a dropdown to the GUI."""
        assert len(options) > 0
        if initial_value is None:
            initial_value = options[0]
        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={
                "value": initial_value,
                "options": options,
                "disabled": disabled,
            },
        )

    def add_gui_slider(
        self,
        name: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: Optional[IntOrFloat],
        initial_value: IntOrFloat,
        disabled: bool = False,
    ) -> GuiHandle[IntOrFloat]:
        """Add a dropdown to the GUI."""
        assert max >= min
        if step is not None:
            assert step <= (max - min)
        assert max >= initial_value >= min

        return _add_gui_impl(
            self,
            name,
            initial_value,
            leva_conf={
                "value": initial_value,
                "min": min,
                "max": max,
                "step": step,
                "disabled": disabled,
            },
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

    @_wrap_message(_messages.MeshMessage)
    def add_mesh(self):
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
            cb(client_id, message)

    @abc.abstractmethod
    def _queue(self, message: _messages.Message) -> None:
        """Abstract method for sending messages."""
        ...


def _handle_gui_updates(
    self: MessageApi,
    client_id: ClientId,
    message: _messages.Message,
) -> None:
    if not isinstance(message, _messages.GuiUpdateMessage):
        return

    handle_state = self._handle_state_from_gui_name.get(message.name, None)
    if handle_state is None:
        return

    # Only call update when value has actually changed.
    if not handle_state.is_button and message.value == handle_state.value:
        return

    # Update state.
    handle_state.value = handle_state.typ(message.value)
    handle_state.last_updated = time.time()

    # Trigger callbacks.
    for cb in self._handle_state_from_gui_name[message.name].update_cb:
        cb(message.value)
    if handle_state.sync_cb is not None:
        handle_state.sync_cb(client_id, message.value)


def _add_gui_impl(
    api: MessageApi,
    name: str,
    initial_value: Any,
    leva_conf: dict,
    is_button: bool = False,
) -> GuiHandle[Any]:
    """Private helper for adding a simple GUI element."""

    handle = _GuiHandleState(
        name,
        typ=type(initial_value),
        api=api,
        value=initial_value,
        last_updated=time.time(),
        folder_label=api._gui_folder_label,
        update_cb=[],
        leva_conf=leva_conf,
        is_button=is_button,
    )
    api._handle_state_from_gui_name[name] = handle
    handle.cleanup_cb = lambda: api._handle_state_from_gui_name.pop(name)

    # For broadcasted GUI handles, we should synchronize all clients.
    from ._server import ViserServer

    if not is_button and isinstance(api, ViserServer):

        def sync_other_clients(client_id: ClientId, value: Any) -> None:
            message = _messages.GuiSetMessage(name=name, value=value)
            message.excluded_self_client = client_id
            api._queue(message)

        handle.sync_cb = sync_other_clients

    api._queue(
        _messages.GuiAddMessage(
            name=name,
            folder=api._gui_folder_label,
            leva_conf=leva_conf,
        )
    )
    return GuiHandle(handle)
