from __future__ import annotations

import base64
import dataclasses
import re
import time
import uuid
import warnings
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Literal,
    Tuple,
    TypeVar,
    cast,
    get_type_hints,
)

import imageio.v3 as iio
import numpy as np
from typing_extensions import Protocol

from . import _messages
from ._icons import svg_from_icon
from ._icons_enum import IconName
from ._messages import (
    GuiBaseProps,
    GuiButtonGroupProps,
    GuiCheckboxProps,
    GuiCloseModalMessage,
    GuiDropdownProps,
    GuiFolderProps,
    GuiMarkdownProps,
    GuiMultiSliderProps,
    GuiNumberProps,
    GuiPlotlyProps,
    GuiProgressBarProps,
    GuiRemoveMessage,
    GuiRgbaProps,
    GuiRgbProps,
    GuiSliderProps,
    GuiTabGroupProps,
    GuiTextProps,
    GuiUpdateMessage,
    GuiVector2Props,
    GuiVector3Props,
)
from ._scene_api import _encode_image_binary
from .infra import ClientId

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ._gui_api import GuiApi
    from ._viser import ClientHandle


T = TypeVar("T")
TGuiHandle = TypeVar("TGuiHandle", bound="_GuiInputHandle")


def _make_uuid() -> str:
    """Return a unique ID for referencing GUI elements."""
    return str(uuid.uuid4())


class GuiContainerProtocol(Protocol):
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )


class SupportsRemoveProtocol(Protocol):
    def remove(self) -> None: ...


class GuiPropsProtocol(Protocol):
    order: float


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    uuid: str
    gui_api: GuiApi
    value: T
    props: GuiPropsProtocol
    parent_container_id: str
    """Container that this GUI input was placed into."""

    update_timestamp: float = 0.0
    update_cb: list[Callable[[GuiEvent], None]] = dataclasses.field(
        default_factory=list
    )
    """Registered functions to call when this input is updated."""

    is_button: bool = False
    """Indicates a button element, which requires special handling."""

    sync_cb: Callable[[ClientId, dict[str, Any]], None] | None = None
    """Callback for synchronizing inputs across clients."""

    removed: bool = False


class _OverridableGuiPropApi:
    """Mixin that allows reading/assigning properties defined in each scene node message."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_impl":
            return object.__setattr__(self, name, value)

        handle = cast(_GuiInputHandle, self)
        # Get the value of the T TypeVar.
        if name in self._prop_hints:
            if getattr(handle._impl.props, name) == value:
                # Do nothing. Assumes equality is defined for the prop value.
                return
            setattr(handle._impl.props, name, value)
            handle._impl.gui_api._websock_interface.queue_message(
                _messages.GuiUpdateMessage(handle._impl.uuid, {name: value})
            )
        else:
            return object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._prop_hints:
            return getattr(self._impl.props, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    @cached_property
    def _prop_hints(self) -> Dict[str, Any]:
        return get_type_hints(type(self._impl.props))


class _GuiHandle(
    Generic[T],
    _OverridableGuiPropApi if not TYPE_CHECKING else object,
):
    # Let's shove private implementation details in here...
    def __init__(self, _impl: _GuiHandleState[T]) -> None:
        self._impl = _impl
        parent = self._impl.gui_api._container_handle_from_uuid[
            self._impl.parent_container_id
        ]
        parent._children[self._impl.uuid] = self

        if isinstance(self, _GuiInputHandle):
            self._impl.gui_api._gui_input_handle_from_uuid[self._impl.uuid] = self

    def remove(self) -> None:
        """Permanently remove this GUI element from the visualizer."""

        # Warn if already removed.
        if self._impl.removed:
            warnings.warn(
                f"Attempted to remove an already removed {self.__class__.__name__}.",
                stacklevel=2,
            )
            return
        self._impl.removed = True

        # Send remove to client(s) + update internal state.
        gui_api = self._impl.gui_api
        gui_api._websock_interface.get_message_buffer().remove_messages(
            # Don't send outdated GUI updates to new clients.
            # This is brittle...
            lambda message: getattr(message, "uuid") == self._impl.uuid
        )
        gui_api._websock_interface.queue_message(GuiRemoveMessage(self._impl.uuid))
        parent = gui_api._container_handle_from_uuid[self._impl.parent_container_id]
        parent._children.pop(self._impl.uuid)

        if isinstance(self, _GuiInputHandle):
            gui_api._gui_input_handle_from_uuid.pop(self._impl.uuid)


class _GuiInputHandle(
    _GuiHandle[T],
    Generic[T],
    GuiBaseProps,
):
    @property
    def value(self) -> T:
        """Value of the GUI input. Synchronized automatically when assigned.

        :meta private:
        """
        # ^Note: we mark this property as private for Sphinx because I haven't
        # been able to get it to resolve the TypeVar in a readable way.
        # For the documentation's sake, we'll be manually adding ::attribute directives below.
        return self._impl.value

    @value.setter
    def value(self, value: T | np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            assert len(value.shape) <= 1, f"{value.shape} should be at most 1D!"
            value = tuple(map(float, value))  # type: ignore

        # Send to client, except for buttons.
        if not self._impl.is_button:
            self._impl.gui_api._websock_interface.queue_message(
                GuiUpdateMessage(self._impl.uuid, {"value": value})
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


StringType = TypeVar("StringType", bound=str)


# GuiInputHandle[T] is used for all inputs except for buttons.
#
# We inherit from _GuiInputHandle to special-case buttons because the usage semantics
# are slightly different: we have `on_click()` instead of `on_update()`.
class GuiInputHandle(_GuiInputHandle[T], Generic[T]):
    """A handle is created for each GUI element that is added in `viser`.
    Handles can be used to read and write state.

    When a GUI element is added via :attr:`ViserServer.gui`, state is
    synchronized between all connected clients. When a GUI element is added via
    :attr:`ClientHandle.gui`, state is local to a specific client.
    """

    def on_update(
        self: TGuiHandle, func: Callable[[GuiEvent[TGuiHandle]], Any]
    ) -> Callable[[GuiEvent[TGuiHandle]], None]:
        """Attach a function to call when a GUI input is updated. Callbacks stack (need
        to be manually removed via :meth:`remove_update_callback()`) and will be called
        from a thread."""
        self._impl.update_cb.append(func)
        return func

    def remove_update_callback(
        self, callback: Literal["all"] | Callable = "all"
    ) -> None:
        """Remove update callbacks from the GUI input.

        Args:
            callback: Either "all" to remove all callbacks, or a specific callback function to remove.
        """
        if callback == "all":
            self._impl.update_cb.clear()
        else:
            self._impl.update_cb = [cb for cb in self._impl.update_cb if cb != callback]


class GuiCheckboxHandle(GuiInputHandle[bool], GuiCheckboxProps):
    """Handle for checkbox inputs.

    .. attribute:: value
       :type: bool

       Value of the input. Synchronized automatically when assigned.
    """


class GuiTextHandle(GuiInputHandle[str], GuiTextProps):
    """Handle for text inputs.

    .. attribute:: value
       :type: str

       Value of the input. Synchronized automatically when assigned.
    """


IntOrFloat = TypeVar("IntOrFloat", int, float)


class GuiNumberHandle(GuiInputHandle[IntOrFloat], Generic[IntOrFloat], GuiNumberProps):
    """Handle for number inputs.

    .. attribute:: value
       :type: IntOrFloat

       Value of the input. Synchronized automatically when assigned.
    """


class GuiSliderHandle(GuiInputHandle[IntOrFloat], Generic[IntOrFloat], GuiSliderProps):
    """Handle for slider inputs.

    .. attribute:: value
       :type: IntOrFloat

       Value of the input. Synchronized automatically when assigned.
    """


class GuiMultiSliderHandle(
    GuiInputHandle[Tuple[IntOrFloat, ...]], Generic[IntOrFloat], GuiMultiSliderProps
):
    """Handle for multi-slider inputs.

    .. attribute:: value
       :type: tuple[IntOrFloat, ...]

       Value of the input. Synchronized automatically when assigned.
    """


class GuiRgbHandle(GuiInputHandle[Tuple[int, int, int]], GuiRgbProps):
    """Handle for RGB color inputs.

    .. attribute:: value
       :type: tuple[int, int, int]

       Value of the input. Synchronized automatically when assigned.
    """


class GuiRgbaHandle(GuiInputHandle[Tuple[int, int, int, int]], GuiRgbaProps):
    """Handle for RGBA color inputs.

    .. attribute:: value
       :type: tuple[int, int, int, int]

       Value of the input. Synchronized automatically when assigned.
    """


class GuiVector2Handle(GuiInputHandle[Tuple[float, float]], GuiVector2Props):
    """Handle for 2D vector inputs.

    .. attribute:: value
       :type: tuple[float, float]

       Value of the input. Synchronized automatically when assigned.
    """


class GuiVector3Handle(GuiInputHandle[Tuple[float, float, float]], GuiVector3Props):
    """Handle for 3D vector inputs.

    .. attribute:: value
       :type: tuple[float, float, float]

       Value of the input. Synchronized automatically when assigned.
    """


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


class GuiButtonHandle(_GuiInputHandle[bool]):
    """Handle for a button input in our visualizer.

    .. attribute:: value
       :type: bool

       Value of the button. Set to `True` when the button is pressed. Can be manually set back to `False`.
    """

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


class GuiUploadButtonHandle(_GuiInputHandle[UploadedFile]):
    """Handle for an upload file button in our visualizer.

    The `.value` attribute will be updated with the contents of uploaded files.

    .. attribute:: value
       :type: UploadedFile

       Value of the input. Contains information about the uploaded file.
    """

    def on_upload(
        self: TGuiHandle, func: Callable[[GuiEvent[TGuiHandle]], None]
    ) -> Callable[[GuiEvent[TGuiHandle]], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        self._impl.update_cb.append(func)
        return func


class GuiButtonGroupHandle(_GuiInputHandle[str], GuiButtonGroupProps):
    """Handle for a button group input in our visualizer.

    .. attribute:: value
       :type: str

       Value of the input. Represents the currently selected button in the group.
    """

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
    def disabled(self, disabled: bool) -> None:  # type: ignore
        """Button groups cannot be disabled."""
        assert not disabled, "Button groups cannot be disabled."


class GuiDropdownHandle(
    GuiInputHandle[StringType], Generic[StringType], GuiDropdownProps
):
    """Handle for a dropdown-style GUI input in our visualizer.

    .. attribute:: value
       :type: StringType

       Value of the input. Represents the currently selected option in the dropdown.
    """

    @property
    def options(self) -> tuple[StringType, ...]:
        """Options for our dropdown. Synchronized automatically when assigned.

        For projects that care about typing: the static type of `options` should be
        consistent with the `StringType` associated with a handle. Literal types will be
        inferred where possible when handles are instantiated; for the most flexibility,
        we can declare handles as `GuiDropdownHandle[str]`.
        """
        assert isinstance(self._impl.props, GuiDropdownProps)
        return self._impl.props.options  # type: ignore

    @options.setter
    def options(self, options: Iterable[StringType]) -> None:  # type: ignore
        assert isinstance(self._impl.props, GuiDropdownProps)
        options = tuple(options)
        self._impl.props.options = options

        need_to_overwrite_value = self.value not in options
        if need_to_overwrite_value:
            self._impl.gui_api._websock_interface.queue_message(
                GuiUpdateMessage(
                    self._impl.uuid,
                    {"options": options, "value": options},
                )
            )
            self._impl.value = options[0]
        else:
            self._impl.gui_api._websock_interface.queue_message(
                GuiUpdateMessage(
                    self._impl.uuid,
                    {"options": options},
                )
            )


class GuiTabGroupHandle(_GuiHandle[None], GuiTabGroupProps):
    """Handle for a tab group. Call :meth:`add_tab()` to add a tab."""

    def __init__(self, _impl: _GuiHandleState[None]) -> None:
        super().__init__(_impl=_impl)
        self._tab_handles: list[GuiTabHandle] = []

    def add_tab(self, label: str, icon: IconName | None = None) -> GuiTabHandle:
        """Add a tab. Returns a handle we can use to add GUI elements to it."""

        uuid = _make_uuid()

        # We may want to make this thread-safe in the future.
        out = GuiTabHandle(_parent=self, _id=uuid)

        self._tab_handles.append(out)
        self._tab_labels = self._tab_labels + (label,)
        self._tab_icons_html = self._tab_icons_html + (
            None if icon is None else svg_from_icon(icon),
        )
        self._tab_container_ids = tuple(handle._id for handle in self._tab_handles)
        return out

    def __post_init__(self) -> None:
        parent = self._impl.gui_api._container_handle_from_uuid[
            self._impl.parent_container_id
        ]
        parent._children[self._impl.uuid] = self

    def remove(self) -> None:
        """Remove this tab group and all contained GUI elements."""
        # Warn if already removed.
        if self._impl.removed:
            warnings.warn(
                f"Attempted to remove an already removed {self.__class__.__name__}.",
                stacklevel=2,
            )
            return
        self._impl.removed = True

        # Remove tabs, then self.
        for tab in tuple(self._tab_handles):
            tab.remove()
        gui_api = self._impl.gui_api
        gui_api._websock_interface.get_message_buffer().remove_messages(
            # Don't send outdated GUI updates to new clients.
            lambda message: isinstance(message, GuiUpdateMessage)
            and message.uuid == self._impl.uuid
        )
        gui_api._websock_interface.queue_message(GuiRemoveMessage(self._impl.uuid))
        parent = gui_api._container_handle_from_uuid[self._impl.parent_container_id]
        parent._children.pop(self._impl.uuid)


@dataclasses.dataclass
class GuiTabHandle:
    """Use as a context to place GUI elements into a tab."""

    _parent: GuiTabGroupHandle
    _id: str  # Used as container ID of children.
    _container_id_restore: str | None = None
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )
    _removed: bool = False

    def __enter__(self) -> GuiTabHandle:
        self._container_id_restore = self._parent._impl.gui_api._get_container_uid()
        self._parent._impl.gui_api._set_container_uid(self._id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._parent._impl.gui_api._set_container_uid(self._container_id_restore)
        self._container_id_restore = None

    def __post_init__(self) -> None:
        self._parent._impl.gui_api._container_handle_from_uuid[self._id] = self

    def remove(self) -> None:
        """Permanently remove this tab and all contained GUI elements from the
        visualizer."""
        # Warn if already removed.
        if self._removed:
            warnings.warn(
                f"Attempted to remove an already removed {self.__class__.__name__}.",
                stacklevel=2,
            )
            return
        self._removed = True

        # We may want to make this thread-safe in the future.
        found_index = -1
        for i, tab in enumerate(self._parent._tab_handles):
            if tab is self:
                found_index = i
                break
        assert found_index != -1, "Tab already removed!"

        self._parent._tab_labels = (
            self._parent._tab_labels[:found_index]
            + self._parent._tab_labels[found_index + 1 :]
        )
        self._parent._tab_icons_html = (
            self._parent._tab_icons_html[:found_index]
            + self._parent._tab_icons_html[found_index + 1 :]
        )
        self._parent._tab_handles = (
            self._parent._tab_handles[:found_index]
            + self._parent._tab_handles[found_index + 1 :]
        )

        for child in tuple(self._children.values()):
            child.remove()
        self._parent._impl.gui_api._container_handle_from_uuid.pop(self._id)


class GuiFolderHandle(_GuiHandle, GuiFolderProps):
    """Use as a context to place GUI elements into a folder."""

    def __init__(self, _impl: _GuiHandleState[None]) -> None:
        super().__init__(_impl=_impl)
        self._impl.gui_api._container_handle_from_uuid[self._impl.uuid] = self
        self._children = {}
        parent = self._impl.gui_api._container_handle_from_uuid[
            self._impl.parent_container_id
        ]
        parent._children[self._impl.uuid] = self

    def __enter__(self) -> GuiFolderHandle:
        self._container_id_restore = self._impl.gui_api._get_container_uid()
        self._impl.gui_api._set_container_uid(self._impl.uuid)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._impl.gui_api._set_container_uid(self._container_id_restore)
        self._container_id_restore = None

    def remove(self) -> None:
        """Permanently remove this folder and all contained GUI elements from the
        visualizer."""
        # Warn if already removed.
        if self._impl.removed:
            warnings.warn(
                f"Attempted to remove an already removed {self.__class__.__name__}.",
                stacklevel=2,
            )
            return
        self._impl.removed = True

        # Remove children, then self.
        gui_api = self._impl.gui_api
        gui_api._websock_interface.get_message_buffer().remove_messages(
            # Don't send outdated GUI updates to new clients.
            lambda message: isinstance(message, GuiUpdateMessage)
            and message.uuid == self._impl.uuid
        )
        gui_api._websock_interface.queue_message(GuiRemoveMessage(self._impl.uuid))
        for child in tuple(self._children.values()):
            child.remove()
        parent = gui_api._container_handle_from_uuid[self._impl.parent_container_id]
        parent._children.pop(self._impl.uuid)
        gui_api._container_handle_from_uuid.pop(self._impl.uuid)


@dataclasses.dataclass
class GuiModalHandle:
    """Use as a context to place GUI elements into a modal."""

    _gui_api: GuiApi
    _uid: str  # Used as container ID of children.
    _container_uid_restore: str | None = None
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )

    def __enter__(self) -> GuiModalHandle:
        self._container_uid_restore = self._gui_api._get_container_uid()
        self._gui_api._set_container_uid(self._uid)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_uid_restore is not None
        self._gui_api._set_container_uid(self._container_uid_restore)
        self._container_uid_restore = None

    def __post_init__(self) -> None:
        self._gui_api._container_handle_from_uuid[self._uid] = self

    def close(self) -> None:
        """Close this modal and permananently remove all contained GUI elements."""
        self._gui_api._websock_interface.queue_message(
            GuiCloseModalMessage(self._uid),
        )
        for child in tuple(self._children.values()):
            child.remove()
        self._gui_api._container_handle_from_uuid.pop(self._uid)


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
        media_type, binary = _encode_image_binary(image, "png")
        url = base64.b64encode(binary).decode("utf-8")
        return f"data:{media_type};base64,{url}"
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


class GuiProgressBarHandle(_GuiInputHandle[float], GuiProgressBarProps):
    """Handle for updating and removing progress bars."""


class GuiMarkdownHandle(_GuiHandle[None], GuiMarkdownProps):
    """Handling for updating and removing markdown elements."""

    def __init__(self, _impl: _GuiHandleState, _content: str, _image_root: Path | None):
        super().__init__(_impl=_impl)
        self._content = _content
        self._image_root = _image_root

    @property
    def content(self) -> str:
        """Current content of this markdown element. Synchronized automatically when assigned."""
        assert self._content is not None
        return self._content

    @content.setter
    def content(self, content: str) -> None:
        self._content = content
        self._markdown = _parse_markdown(content, self._image_root)


class GuiPlotlyHandle(_GuiHandle[None], GuiPlotlyProps):
    """Handle for updating and removing Plotly figures."""

    def __init__(self, _impl: _GuiHandleState, _figure: go.Figure):
        super().__init__(_impl=_impl)
        self._figure = _figure

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
        self._plotly_json_str = json_str
