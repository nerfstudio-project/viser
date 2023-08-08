from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as onp

from ._icons import base64_from_icon
from ._icons_enum import Icon
from ._messages import (
    GuiAddDropdownMessage,
    GuiAddTabGroupMessage,
    GuiRemoveContainerChildrenMessage,
    GuiRemoveMessage,
    GuiSetDisabledMessage,
    GuiSetValueMessage,
    GuiSetVisibleMessage,
)
from .infra import ClientId

if TYPE_CHECKING:
    from ._gui_api import GuiApi


T = TypeVar("T")
TGuiHandle = TypeVar("TGuiHandle", bound="_GuiHandle")


def _make_unique_id() -> str:
    """Return a unique ID for referencing GUI elements."""
    return str(uuid.uuid4())


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    label: str
    typ: Type[T]
    container: GuiApi
    value: T
    update_timestamp: float

    container_id: str
    """Container that this GUI input was placed into."""

    update_cb: List[Callable[[Any], None]]
    """Registered functions to call when this input is updated."""

    is_button: bool
    """Indicates a button element, which requires special handling."""

    sync_cb: Optional[Callable[[ClientId, T], None]]
    """Callback for synchronizing inputs across clients."""

    cleanup_cb: Optional[Callable[[], Any]]
    """Function to call when GUI element is removed."""

    disabled: bool
    visible: bool

    order: float
    id: str
    initial_value: T
    hint: Optional[str]


@dataclasses.dataclass
class _GuiHandle(Generic[T]):
    # Let's shove private implementation details in here...
    _impl: _GuiHandleState[T]

    # Should we use @property for get_value / set_value, set_hidden, etc?
    #
    # Benefits:
    #   @property is syntactically very nice.
    #   `gui.value = ...` is really tempting!
    #   Feels a bit more magical.
    #
    # Downsides:
    #   Consistency: not everything that can be written can be read, and not everything
    #   that can be read can be written. `get_`/`set_` makes this really clear.
    #   Clarity: some things that we read (like client mappings) are copied before
    #   they're returned. An attribute access obfuscates the overhead here.
    #   Flexibility: getter/setter types should match. https://github.com/python/mypy/issues/3004
    #   Feels a bit more magical.
    #
    # Is this worth the tradeoff?

    @property
    def value(self) -> T:
        """Value of the GUI input. Synchronized automatically when assigned."""
        return self._impl.value

    @value.setter
    def value(self, value: Union[T, onp.ndarray]) -> None:
        if isinstance(value, onp.ndarray):
            assert len(value.shape) <= 1, f"{value.shape} should be at most 1D!"
            value = tuple(map(float, value))  # type: ignore

        # Send to client, except for buttons.
        if not self._impl.is_button:
            self._impl.container._get_api()._queue(
                GuiSetValueMessage(self._impl.id, value)  # type: ignore
            )

        # Set internal state. We automatically convert numpy arrays to the expected
        # internal type. (eg 1D arrays to tuples)
        self._impl.value = type(self._impl.value)(value)  # type: ignore
        self._impl.update_timestamp = time.time()

        # Call update callbacks.
        for cb in self._impl.update_cb:
            # Pushing callbacks into separate threads helps prevent deadlocks when we
            # have a lock in a callback. TODO: revisit other callbacks.
            threading.Thread(target=lambda: cb(self)).start()

    @property
    def update_timestamp(self) -> float:
        """Get the last time that this input was updated."""
        return self._impl.update_timestamp

    @property
    def disabled(self) -> bool:
        """Allow/disallow user interaction with the input. Synchronized automatically
        when assigned."""
        return self._impl.disabled

    @disabled.setter
    def disabled(self, disabled: bool) -> None:
        if disabled == self.disabled:
            return

        self._impl.container._get_api()._queue(
            GuiSetDisabledMessage(self._impl.id, disabled=disabled)
        )
        self._impl.disabled = disabled

    @property
    def visible(self) -> bool:
        """Temporarily show or hide this GUI element from the visualizer. Synchronized
        automatically when assigned."""
        return self._impl.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self.visible:
            return

        self._impl.container._get_api()._queue(
            GuiSetVisibleMessage(self._impl.id, visible=visible)
        )
        self._impl.visible = visible

    def remove(self) -> None:
        """Permanently remove this GUI element from the visualizer."""
        self._impl.container._get_api()._queue(GuiRemoveMessage(self._impl.id))
        assert self._impl.cleanup_cb is not None
        self._impl.cleanup_cb()


StringType = TypeVar("StringType", bound=str)


@dataclasses.dataclass
class GuiHandle(_GuiHandle[T], Generic[T]):
    """Handle for a general GUI inputs in our visualizer.

    Lets us get values, set values, and detect updates."""

    def on_update(
        self: TGuiHandle, func: Callable[[TGuiHandle], None]
    ) -> Callable[[TGuiHandle], None]:
        """Attach a function to call when a GUI input is updated. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func


@dataclasses.dataclass
class GuiButtonHandle(_GuiHandle[bool]):
    """Handle for a button input in our visualizer.

    Lets us detect clicks."""

    def on_click(
        self: TGuiHandle, func: Callable[[TGuiHandle], None]
    ) -> Callable[[TGuiHandle], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func


@dataclasses.dataclass
class GuiButtonGroupHandle(_GuiHandle[StringType], Generic[StringType]):
    """Handle for a button group input in our visualizer.

    Lets us detect clicks."""

    def on_click(
        self: TGuiHandle, func: Callable[[TGuiHandle], None]
    ) -> Callable[[TGuiHandle], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func

    @property
    def disabled(self) -> bool:
        """Button groups cannot be disabled."""
        return False

    @disabled.setter
    def disabled(self, disabled: bool) -> None:
        """Button groups cannot be disabled."""
        assert not disabled, "Button groups cannot be disabled."


@dataclasses.dataclass
class GuiDropdownHandle(GuiHandle[StringType], Generic[StringType]):
    """Handle for a dropdown-style GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""

    _impl_options: Tuple[StringType, ...]

    @property
    def options(self) -> Tuple[StringType, ...]:
        """Options for our dropdown. Synchronized automatically when assigned.

        For projects that care about typing: the static type of `options` should be
        consistent with the `StringType` associated with a handle. Literal types will be
        inferred where possible when handles are instantiated; for the most flexibility,
        we can declare handles as `_GuiHandle[str]`.
        """
        return self._impl_options

    @options.setter
    def options(self, options: Iterable[StringType]) -> None:
        self._impl_options = tuple(options)
        if self._impl.initial_value not in self._impl_options:
            self._impl.initial_value = self._impl_options[0]

        self._impl.container._get_api()._queue(
            GuiAddDropdownMessage(
                order=self._impl.order,
                id=self._impl.id,
                label=self._impl.label,
                container_id=self._impl.container_id,
                hint=self._impl.hint,
                initial_value=self._impl.initial_value,
                options=self._impl_options,
            )
        )

        if self.value not in self._impl_options:
            self.value = self._impl_options[0]


@dataclasses.dataclass(frozen=True)
class GuiTabGroupHandle:
    _tab_group_id: str
    _labels: List[str]
    _icons_base64: List[Optional[str]]
    _tab_container_ids: List[str]
    _gui_api: GuiApi
    _container_id: str

    def add_tab(self, label: str, icon: Optional[Icon] = None) -> GuiTabHandle:
        """Add a tab. Returns a handle we can use to add GUI elements to it."""

        id = _make_unique_id()

        # We may want to make this thread-safe in the future.
        self._labels.append(label)
        self._icons_base64.append(None if icon is None else base64_from_icon(icon))
        self._tab_container_ids.append(id)

        self._sync_with_client()

        return GuiTabHandle(_parent=self, _container_id=id)

    def remove(self) -> None:
        """Remove this tab group and all contained GUI elements."""
        self._gui_api._get_api()._queue(GuiRemoveMessage(self._tab_group_id))
        # Containers will be removed automatically by the client.
        #
        # for tab_container_id in self._tab_container_ids:
        #     self._gui_api._get_api()._queue(
        #         _messages.GuiRemoveContainerChildrenMessage(tab_container_id)
        #     )

    def _sync_with_client(self) -> None:
        """Send a message that syncs tab state with the client."""
        self._gui_api._get_api()._queue(
            GuiAddTabGroupMessage(
                order=time.time(),
                id=self._tab_group_id,
                container_id=self._container_id,
                tab_labels=tuple(self._labels),
                tab_icons_base64=tuple(self._icons_base64),
                tab_container_ids=tuple(self._tab_container_ids),
            )
        )


@dataclasses.dataclass
class GuiFolderHandle:
    """Use as a context to place GUI elements into a folder."""

    _gui_api: GuiApi
    _container_id: str
    _container_id_restore: Optional[str] = None

    def __enter__(self) -> None:
        self._container_id_restore = self._gui_api._get_container_id()
        self._gui_api._set_container_id(self._container_id)

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def remove(self) -> None:
        """Permanently remove this folder and all contained GUI elements from the
        visualizer."""
        self._gui_api._get_api()._queue(GuiRemoveMessage(self._container_id))


@dataclasses.dataclass
class GuiTabHandle:
    """Use as a context to place GUI elements into a tab."""

    _parent: GuiTabGroupHandle
    _container_id: str
    _container_id_restore: Optional[str] = None

    def __enter__(self) -> None:
        self._container_id_restore = self._parent._gui_api._get_container_id()
        self._parent._gui_api._set_container_id(self._container_id)

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._parent._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def remove(self) -> None:
        """Permanently remove this tab and all contained GUI elements from the
        visualizer."""
        # We may want to make this thread-safe in the future.
        container_index = self._parent._tab_container_ids.index(self._container_id)
        assert container_index != -1, "Tab already removed!"

        # Container needs to be manually removed.
        self._parent._gui_api._get_api()._queue(
            GuiRemoveContainerChildrenMessage(self._container_id)
        )

        self._parent._labels.pop(container_index)
        self._parent._icons_base64.pop(container_index)
        self._parent._tab_container_ids.pop(container_index)

        self._parent._sync_with_client()


@dataclasses.dataclass
class GuiMarkdownHandle:
    """Use to remove markdown."""

    _gui_api: GuiApi
    _id: str
    _visible: bool

    @property
    def visible(self) -> bool:
        """Temporarily show or hide this GUI element from the visualizer. Synchronized
        automatically when assigned."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self.visible:
            return

        self._gui_api._get_api()._queue(GuiSetVisibleMessage(self._id, visible=visible))
        self._visible = visible

    def remove(self) -> None:
        """Permanently remove this markdown from the visualizer."""
        self._gui_api._get_api()._queue(GuiRemoveMessage(self._id))
