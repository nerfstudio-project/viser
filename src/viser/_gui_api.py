# mypy: disable-error-code="misc"
#
# We suppress overload errors that depend on LiteralString support.
# - https://github.com/python/mypy/issues/12554
from __future__ import annotations


from dataclasses import field, InitVar
from functools import wraps
import time
from typing import Optional, Literal, Union, TypeVar, Generic, Tuple, Type
from typing import Callable, Any
from dataclasses import dataclass
try:
    from typing import Concatenate
except ImportError:
    from typing_extensions import Concatenate
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec
from ._gui_components import GuiApiMixin


import abc
import dataclasses
import threading
import time
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import numpy as onp
from typing_extensions import Literal, LiteralString

from . import _messages
from ._gui_handles import (
    GuiButtonGroupHandle,
    GuiButtonHandle,
    GuiContainerProtocol,
    GuiDropdownHandle,
    GuiEvent,
    GuiFolderHandle,
    GuiInputHandle,
    GuiMarkdownHandle,
    GuiModalHandle,
    GuiTabGroupHandle,
    SupportsRemoveProtocol,
    _GuiHandleState,
    _GuiInputHandle,
    _make_unique_id,
)
from ._icons import base64_from_icon
from ._icons_enum import Icon
from ._message_api import MessageApi, cast_vector
from ._gui_components import Property

if TYPE_CHECKING:
    from .infra import ClientId

IntOrFloat = TypeVar("IntOrFloat", int, float)
TString = TypeVar("TString", bound=str)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)
TProps = TypeVar("TProps")
TReturn = TypeVar('TReturn')
TArgs = ParamSpec('TArgs')
T = TypeVar("T")


def _compute_step(x: Optional[float]) -> float:  # type: ignore
    """For number inputs: compute an increment size from some number.

    Example inputs/outputs:
        100 => 1
        12 => 1
        12.1 => 0.1
        12.02 => 0.01
        0.004 => 0.001
    """
    return 1 if x is None else 10 ** (-_compute_precision_digits(x))


def _compute_precision_digits(x: float) -> int:
    """For number inputs: compute digits of precision from some number.

    Example inputs/outputs:
        100 => 0
        12 => 0
        12.1 => 1
        10.2 => 1
        0.007 => 3
    """
    digits = 0
    while x != round(x, ndigits=digits) and digits < 7:
        digits += 1
    return digits


@dataclasses.dataclass
class _RootGuiContainer:
    _children: Dict[str, SupportsRemoveProtocol]


_global_order_counter = 0


def _apply_default_order(order: Optional[float]) -> float:
    """Apply default ordering logic for GUI elements.

    If `order` is set to a float, this function is a no-op and returns it back.
    Otherwise, we increment and return the value of a global counter.
    """
    if order is not None:
        return order

    global _global_order_counter
    _global_order_counter += 1
    return _global_order_counter



class ComponentHandle(Generic[TProps]):
    _id: str
    _props: TProps
    _api_update: Callable[[str, dict], None]
    _update_timestamp: float

    def __init__(self, update: Callable[[str, Dict[str, Any]]], id: str, props: TProps):
        self._id = id
        self._props = props
        self._api_update = update
        self._update_timestamp = time.time()

    @property
    def id(self):
        return self._id

    def _update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self._props, k):
                raise AttributeError(f"Component has no property {k}")
            setattr(self._props, k, v)
        self._update_timestamp = time.time()

        # Raise message to update component.
        self._api_update(self.id, kwargs)

    def property(self, name: str) -> Property[T]:
        props = object.__getattribute__(self, "_props")
        update = object.__getattribute__(self, "_update")
        if not hasattr(props, name):
            raise AttributeError(f"Component has no property {name}")
        return Property(
            lambda: getattr(props, name),
            lambda value: update(**{name: value}),
        )

    def __getattr__(self, name: str) -> T:
        if not hasattr(ComponentHandle, name):
            props = object.__getattribute__(self, "_props")
            if hasattr(props, name):
                return self.property(name).get()
            else:
                raise AttributeError(f"Component has no property {name}")
        return super().__getattribute__(name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if hasattr(self, "_props"):
            props = object.__getattribute__(self, "_props")
            if hasattr(props, __name):
                return self.property(__name).set(__value)
        return super().__setattr__(__name, __value)





class GuiApi(abc.ABC, GuiApiMixin):
    _target_container_from_thread_id: Dict[int, str] = {}
    """ID of container to put GUI elements into."""

    def __init__(self) -> None:
        super().__init__()

        self._gui_handle_from_id: Dict[str, _GuiInputHandle[Any]] = {}
        self._container_handle_from_id: Dict[str, GuiContainerProtocol] = {
            "root": _RootGuiContainer({})
        }
        self._get_api()._message_handler.register_handler(
            _messages.GuiUpdateMessage, self._handle_gui_updates
        )

    def _handle_gui_updates(
        self, client_id: ClientId, message: _messages.GuiUpdateMessage
    ) -> None:
        """Callback for handling GUI messages."""
        handle = self._gui_handle_from_id.get(message.id, None)
        if handle is None:
            return

        # handle_state = handle._impl

        # # Do some type casting. This is necessary when we expect floats but the
        # # Javascript side gives us integers.
        # if handle_state.typ is tuple:
        #     assert len(message.value) == len(handle_state.value)
        #     value = tuple(
        #         type(handle_state.value[i])(message.value[i])
        #         for i in range(len(message.value))
        #     )
        # else:
        #     value = handle_state.typ(message.value)

        # # Only call update when value has actually changed.
        # if not handle_state.is_button and value == handle_state.value:
        #     return

        # # Update state.
        # with self._get_api()._atomic_lock:
        #     handle_state.value = value
        #     handle_state.update_timestamp = time.time()

        # # Trigger callbacks.
        # for cb in handle_state.update_cb:
        #     from ._viser import ClientHandle, ViserServer

        #     # Get the handle of the client that triggered this event.
        #     api = self._get_api()
        #     if isinstance(api, ClientHandle):
        #         client = api
        #     elif isinstance(api, ViserServer):
        #         client = api.get_clients()[client_id]
        #     else:
        #         assert False

        #     cb(GuiEvent(client, client_id, handle))
        # if handle_state.sync_cb is not None:
        #     handle_state.sync_cb(client_id, value)

    def _get_container_id(self) -> str:
        """Get container ID associated with the current thread."""
        return self._target_container_from_thread_id.get(threading.get_ident(), "root")

    def _set_container_id(self, container_id: str) -> None:
        """Set container ID associated with the current thread."""
        self._target_container_from_thread_id[threading.get_ident()] = container_id

    @abc.abstractmethod
    def _get_api(self) -> MessageApi:
        """Message API to use."""
        ...

    def _update_component_props(self, id: str, kwargs: Dict[str, Any]) -> None:
        """Update properties of a GUI element."""
        self._get_api()._queue(_messages.GuiUpdateMessage(id=id, **kwargs))

    def gui_add_component(self, props: TProps) -> TProps:
        props.order = _apply_default_order(props.order)
        handle = ComponentHandle(self._update_component_props, id=_make_unique_id(), props=props)
        self._get_api()._queue(_messages.GuiAddComponentMessage(
            order=handle.order,
            id=handle.id,
            props=props,
            container_id=self._get_container_id()
        ))
        self._gui_handle_from_id[handle.id] = handle
        return handle
