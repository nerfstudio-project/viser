from __future__ import annotations

import dataclasses
import re
import time
import urllib.parse
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, TypeVar

import imageio.v3 as iio
import numpy as onp
from typing_extensions import Protocol

from ._icons import svg_from_icon
from ._icons_enum import IconName
from ._messages import GuiCloseModalMessage, GuiRemoveMessage, GuiUpdateMessage, Message
from ._scene_api import _encode_image_base64
from .infra import ClientId

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ._gui_api import GuiApi
    from ._viser import ClientHandle


T = TypeVar("T")
TGuiHandle = TypeVar("TGuiHandle", bound="_GuiInputHandle")


def _make_unique_id() -> str:
    """Return a unique ID for referencing GUI elements."""
    return str(uuid.uuid4())


class GuiContainerProtocol(Protocol):
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )


class SupportsRemoveProtocol(Protocol):
    def remove(self) -> None:
        ...


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    label: str
    gui_api: GuiApi
    value: T
    update_timestamp: float

    parent_container_id: str
    """Container that this GUI input was placed into."""

    update_cb: list[Callable[[GuiEvent], None]]
    """Registered functions to call when this input is updated."""

    is_button: bool
    """Indicates a button element, which requires special handling."""

    sync_cb: Callable[[ClientId, dict[str, Any]], None] | None
    """Callback for synchronizing inputs across clients."""

    disabled: bool
    visible: bool

    order: float
    id: str
    hint: str | None

    message_type: type[Message]


@dataclasses.dataclass
class _GuiInputHandle(Generic[T]):
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
    def order(self) -> float:
        """Read-only order value, which dictates the position of the GUI element."""
        return self._impl.order

    @property
    def value(self) -> T:
        """Value of the GUI input. Synchronized automatically when assigned."""
        return self._impl.value

    @value.setter
    def value(self, value: T | onp.ndarray) -> None:
        if isinstance(value, onp.ndarray):
            assert len(value.shape) <= 1, f"{value.shape} should be at most 1D!"
            value = tuple(map(float, value))  # type: ignore

        # Send to client, except for buttons.
        if not self._impl.is_button:
            self._impl.gui_api._websock_interface.queue_message(
                GuiUpdateMessage(self._impl.id, {"value": value})
            )

        # Set internal state. We automatically convert numpy arrays to the expected
        # internal type. (eg 1D arrays to tuples)
        self._impl.value = type(self._impl.value)(value)  # type: ignore
        self._impl.update_timestamp = time.time()

        # Call update callbacks.
        for cb in self._impl.update_cb:
            # Pushing callbacks into separate threads helps prevent deadlocks when we
            # have a lock in a callback. TODO: revisit other callbacks.
            self._impl.gui_api._thread_executor.submit(
                lambda: cb(
                    GuiEvent(
                        client_id=None,
                        client=None,
                        target=self,
                    )
                )
            )

    @property
    def update_timestamp(self) -> float:
        """Read-only timestamp when this input was last updated."""
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

        self._impl.gui_api._websock_interface.queue_message(
            GuiUpdateMessage(self._impl.id, {"disabled": disabled})
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

        self._impl.gui_api._websock_interface.queue_message(
            GuiUpdateMessage(self._impl.id, {"visible": visible})
        )
        self._impl.visible = visible

    def __post_init__(self) -> None:
        """We need to register ourself after construction for callbacks to work."""
        gui_api = self._impl.gui_api

        # TODO: the current way we track GUI handles and children is very manual +
        # error-prone. We should revist this design.
        gui_api._gui_input_handle_from_id[self._impl.id] = self
        parent = gui_api._container_handle_from_id[self._impl.parent_container_id]
        parent._children[self._impl.id] = self

    def remove(self) -> None:
        """Permanently remove this GUI element from the visualizer."""
        gui_api = self._impl.gui_api
        gui_api._websock_interface.queue_message(GuiRemoveMessage(self._impl.id))
        gui_api._gui_input_handle_from_id.pop(self._impl.id)
        parent = gui_api._container_handle_from_id[self._impl.parent_container_id]
        parent._children.pop(self._impl.id)


StringType = TypeVar("StringType", bound=str)


# GuiInputHandle[T] is used for all inputs except for buttons.
#
# We inherit from _GuiInputHandle to special-case buttons because the usage semantics
# are slightly different: we have `on_click()` instead of `on_update()`.
@dataclasses.dataclass
class GuiInputHandle(_GuiInputHandle[T], Generic[T]):
    """A handle is created for each GUI element that is added in `viser`.
    Handles can be used to read and write state.

    When a GUI element is added via :attr:`ViserServer.gui`, state is
    synchronized between all connected clients. When a GUI element is added via
    :attr:`ClientHandle.gui`, state is local to a specific client.
    """

    def on_update(
        self: TGuiHandle, func: Callable[[GuiEvent[TGuiHandle]], None]
    ) -> Callable[[GuiEvent[TGuiHandle]], None]:
        """Attach a function to call when a GUI input is updated. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func


@dataclasses.dataclass(frozen=True)
class GuiEvent(Generic[TGuiHandle]):
    """Information associated with a GUI event, such as an update or click.

    Passed as input to callback functions."""

    client: ClientHandle | None
    """Client that triggered this event."""
    client_id: int | None
    """ID of client that triggered this event."""
    target: TGuiHandle
    """GUI element that was affected."""


@dataclasses.dataclass
class GuiButtonHandle(_GuiInputHandle[bool]):
    """Handle for a button input in our visualizer.

    Lets us detect clicks."""

    def on_click(
        self: TGuiHandle, func: Callable[[GuiEvent[TGuiHandle]], None]
    ) -> Callable[[GuiEvent[TGuiHandle]], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func


@dataclasses.dataclass
class UploadedFile:
    """Result of a file upload."""

    name: str
    """Name of the file."""
    content: bytes
    """Contents of the file."""


@dataclasses.dataclass
class GuiUploadButtonHandle(_GuiInputHandle[UploadedFile]):
    """Handle for an upload file button in our visualizer.

    The `.value` attribute will be updated with the contents of uploaded files.
    """

    def on_upload(
        self: TGuiHandle, func: Callable[[GuiEvent[TGuiHandle]], None]
    ) -> Callable[[GuiEvent[TGuiHandle]], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func


@dataclasses.dataclass
class GuiButtonGroupHandle(_GuiInputHandle[StringType], Generic[StringType]):
    """Handle for a button group input in our visualizer.

    Lets us detect clicks."""

    def on_click(
        self: TGuiHandle, func: Callable[[GuiEvent[TGuiHandle]], None]
    ) -> Callable[[GuiEvent[TGuiHandle]], None]:
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
class GuiDropdownHandle(GuiInputHandle[StringType], Generic[StringType]):
    """Handle for a dropdown-style GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""

    _impl_options: tuple[StringType, ...]

    @property
    def options(self) -> tuple[StringType, ...]:
        """Options for our dropdown. Synchronized automatically when assigned.

        For projects that care about typing: the static type of `options` should be
        consistent with the `StringType` associated with a handle. Literal types will be
        inferred where possible when handles are instantiated; for the most flexibility,
        we can declare handles as `GuiDropdownHandle[str]`.
        """
        return self._impl_options

    @options.setter
    def options(self, options: Iterable[StringType]) -> None:
        self._impl_options = tuple(options)

        need_to_overwrite_value = self.value not in self._impl_options
        if need_to_overwrite_value:
            self._impl.gui_api._websock_interface.queue_message(
                GuiUpdateMessage(
                    self._impl.id,
                    {"options": self._impl_options, "value": self._impl_options[0]},
                )
            )
            self._impl.value = self._impl_options[0]
        else:
            self._impl.gui_api._websock_interface.queue_message(
                GuiUpdateMessage(
                    self._impl.id,
                    {"options": self._impl_options},
                )
            )


@dataclasses.dataclass(frozen=True)
class GuiTabGroupHandle:
    """Handle for a tab group. Call :meth:`add_tab()` to add a tab."""

    _tab_group_id: str
    _labels: list[str]
    _icons_html: list[str | None]
    _tabs: list[GuiTabHandle]
    _gui_api: GuiApi
    _parent_container_id: str
    _order: float

    @property
    def order(self) -> float:
        """Read-only order value, which dictates the position of the GUI element."""
        return self._order

    def add_tab(self, label: str, icon: IconName | None = None) -> GuiTabHandle:
        """Add a tab. Returns a handle we can use to add GUI elements to it."""

        id = _make_unique_id()

        # We may want to make this thread-safe in the future.
        out = GuiTabHandle(_parent=self, _id=id)

        self._labels.append(label)
        self._icons_html.append(None if icon is None else svg_from_icon(icon))
        self._tabs.append(out)

        self._sync_with_client()
        return out

    def __post_init__(self) -> None:
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children[self._tab_group_id] = self

    def remove(self) -> None:
        """Remove this tab group and all contained GUI elements."""
        for tab in tuple(self._tabs):
            tab.remove()
        gui_api = self._gui_api
        gui_api._websock_interface.queue_message(GuiRemoveMessage(self._tab_group_id))
        parent = gui_api._container_handle_from_id[self._parent_container_id]
        parent._children.pop(self._tab_group_id)

    def _sync_with_client(self) -> None:
        """Send messages for syncing tab state with the client."""
        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(
                self._tab_group_id,
                {
                    "tab_labels": tuple(self._labels),
                    "tab_icons_html": tuple(self._icons_html),
                    "tab_container_ids": tuple(tab._id for tab in self._tabs),
                },
            )
        )


@dataclasses.dataclass
class GuiFolderHandle:
    """Use as a context to place GUI elements into a folder."""

    _gui_api: GuiApi
    _id: str  # Used as container ID for children.
    _order: float
    _parent_container_id: str  # Container ID of parent.
    _container_id_restore: str | None = None
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )

    @property
    def order(self) -> float:
        """Read-only order value, which dictates the position of the GUI element."""
        return self._order

    def __enter__(self) -> GuiFolderHandle:
        self._container_id_restore = self._gui_api._get_container_id()
        self._gui_api._set_container_id(self._id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def __post_init__(self) -> None:
        self._gui_api._container_handle_from_id[self._id] = self
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children[self._id] = self

    def remove(self) -> None:
        """Permanently remove this folder and all contained GUI elements from the
        visualizer."""
        self._gui_api._websock_interface.queue_message(GuiRemoveMessage(self._id))
        for child in tuple(self._children.values()):
            child.remove()
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children.pop(self._id)
        self._gui_api._container_handle_from_id.pop(self._id)


@dataclasses.dataclass
class GuiModalHandle:
    """Use as a context to place GUI elements into a modal."""

    _gui_api: GuiApi
    _id: str  # Used as container ID of children.
    _container_id_restore: str | None = None
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )

    def __enter__(self) -> GuiModalHandle:
        self._container_id_restore = self._gui_api._get_container_id()
        self._gui_api._set_container_id(self._id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def __post_init__(self) -> None:
        self._gui_api._container_handle_from_id[self._id] = self

    def close(self) -> None:
        """Close this modal and permananently remove all contained GUI elements."""
        self._gui_api._websock_interface.queue_message(
            GuiCloseModalMessage(self._id),
        )
        for child in tuple(self._children.values()):
            child.remove()
        self._gui_api._container_handle_from_id.pop(self._id)


@dataclasses.dataclass
class GuiTabHandle:
    """Use as a context to place GUI elements into a tab."""

    _parent: GuiTabGroupHandle
    _id: str  # Used as container ID of children.
    _container_id_restore: str | None = None
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )

    def __enter__(self) -> GuiTabHandle:
        self._container_id_restore = self._parent._gui_api._get_container_id()
        self._parent._gui_api._set_container_id(self._id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._parent._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def __post_init__(self) -> None:
        self._parent._gui_api._container_handle_from_id[self._id] = self

    def remove(self) -> None:
        """Permanently remove this tab and all contained GUI elements from the
        visualizer."""
        # We may want to make this thread-safe in the future.
        container_index = -1
        for i, tab in enumerate(self._parent._tabs):
            if tab is self:
                container_index = i
                break
        assert container_index != -1, "Tab already removed!"

        self._parent._labels.pop(container_index)
        self._parent._icons_html.pop(container_index)
        self._parent._tabs.pop(container_index)
        self._parent._sync_with_client()
        for child in tuple(self._children.values()):
            child.remove()
        self._parent._gui_api._container_handle_from_id.pop(self._id)


def _get_data_url(url: str, image_root: Path | None) -> str:
    if not url.startswith("http") and not image_root:
        warnings.warn(
            (
                "No `image_root` provided. All relative paths will be scoped to viser's"
                " installation path."
            ),
            stacklevel=2,
        )
    if url.startswith("http") or url.startswith("data:"):
        return url
    if image_root is None:
        image_root = Path(__file__).parent
    try:
        image = iio.imread(image_root / url)
        data_uri = _encode_image_base64(image, "png")
        url = urllib.parse.quote(f"{data_uri[1]}")
        return f"data:{data_uri[0]};base64,{url}"
    except (IOError, FileNotFoundError):
        warnings.warn(
            f"Failed to read image {url}, with image_root set to {image_root}.",
            stacklevel=2,
        )
        return url


def _parse_markdown(markdown: str, image_root: Path | None) -> str:
    markdown = re.sub(
        r"\!\[([^]]*)\]\(([^]]*)\)",
        lambda match: (
            f"![{match.group(1)}]({_get_data_url(match.group(2), image_root)})"
        ),
        markdown,
    )
    return markdown


@dataclasses.dataclass
class GuiProgressBarHandle:
    """Use to remove markdown."""

    _gui_api: GuiApi
    _id: str
    _visible: bool
    _loading: bool
    _parent_container_id: str  # Parent.
    _order: float
    _value: float

    @property
    def value(self) -> float:
        """Current content of this progress bar element. Synchronized automatically when assigned."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value
        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(
                self._id,
                {"value": value},
            )
        )

    @property
    def loading(self) -> bool:
        """Show this progress bar as loading (animated, striped)."""
        return self._loading

    @loading.setter
    def loading(self, loading: bool) -> None:
        self._loading = loading
        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(
                self._id,
                {"loading": loading},
            )
        )

    @property
    def order(self) -> float:
        """Read-only order value, which dictates the position of the GUI element."""
        return self._order

    @property
    def visible(self) -> bool:
        """Temporarily show or hide this GUI element from the visualizer. Synchronized
        automatically when assigned."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self.visible:
            return

        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(self._id, {"visible": visible})
        )
        self._visible = visible

    def __post_init__(self) -> None:
        """We need to register ourself after construction for callbacks to work."""
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children[self._id] = self

    def remove(self) -> None:
        """Permanently remove this markdown from the visualizer."""
        self._gui_api._websock_interface.queue_message(GuiRemoveMessage(self._id))

        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children.pop(self._id)


@dataclasses.dataclass
class GuiMarkdownHandle:
    """Use to remove markdown."""

    _gui_api: GuiApi
    _id: str
    _visible: bool
    _parent_container_id: str  # Parent.
    _order: float
    _image_root: Path | None
    _content: str | None

    @property
    def content(self) -> str:
        """Current content of this markdown element. Synchronized automatically when assigned."""
        assert self._content is not None
        return self._content

    @content.setter
    def content(self, content: str) -> None:
        self._content = content
        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(
                self._id,
                {"markdown": _parse_markdown(content, self._image_root)},
            )
        )

    @property
    def order(self) -> float:
        """Read-only order value, which dictates the position of the GUI element."""
        return self._order

    @property
    def visible(self) -> bool:
        """Temporarily show or hide this GUI element from the visualizer. Synchronized
        automatically when assigned."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self.visible:
            return

        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(self._id, {"visible": visible})
        )
        self._visible = visible

    def __post_init__(self) -> None:
        """We need to register ourself after construction for callbacks to work."""
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children[self._id] = self

    def remove(self) -> None:
        """Permanently remove this markdown from the visualizer."""
        self._gui_api._websock_interface.queue_message(GuiRemoveMessage(self._id))

        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children.pop(self._id)


@dataclasses.dataclass
class GuiPlotlyHandle:
    """Use to update or remove markdown elements."""

    _gui_api: GuiApi
    _id: str
    _visible: bool
    _parent_container_id: str  # Parent.
    _order: float
    _figure: go.Figure | None
    _aspect: float | None

    @property
    def figure(self) -> go.Figure:
        """Current content of this markdown element. Synchronized automatically when assigned."""
        assert self._figure is not None
        return self._figure

    @figure.setter
    def figure(self, figure: go.Figure) -> None:
        self._figure = figure

        json_str = figure.to_json()
        assert isinstance(json_str, str)

        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(
                self._id,
                {"plotly_json_str": json_str},
            )
        )

    @property
    def aspect(self) -> float:
        """Aspect ratio of the plotly figure, in the control panel."""
        assert self._aspect is not None
        return self._aspect

    @aspect.setter
    def aspect(self, aspect: float) -> None:
        self._aspect = aspect
        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(
                self._id,
                {"aspect": aspect},
            )
        )

    @property
    def order(self) -> float:
        """Read-only order value, which dictates the position of the GUI element."""
        return self._order

    @property
    def visible(self) -> bool:
        """Temporarily show or hide this GUI element from the visualizer. Synchronized
        automatically when assigned."""
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self.visible:
            return

        self._gui_api._websock_interface.queue_message(
            GuiUpdateMessage(self._id, {"visible": visible})
        )
        self._visible = visible

    def __post_init__(self) -> None:
        """We need to register ourself after construction for callbacks to work."""
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children[self._id] = self

    def remove(self) -> None:
        """Permanently remove this markdown from the visualizer."""
        self._gui_api._websock_interface.queue_message(GuiRemoveMessage(self._id))
        parent = self._gui_api._container_handle_from_id[self._parent_container_id]
        parent._children.pop(self._id)
