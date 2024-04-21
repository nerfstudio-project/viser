# mypy: disable-error-code="misc"
#
# We suppress overload errors that depend on LiteralString support.
# - https://github.com/python/mypy/issues/12554
from __future__ import annotations

import abc
import dataclasses
import functools
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
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as onp
from typing_extensions import (
    Literal,
    LiteralString,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
)

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
    GuiPlotlyHandle,
    GuiTabGroupHandle,
    GuiUploadButtonHandle,
    SupportsRemoveProtocol,
    UploadedFile,
    _GuiHandleState,
    _GuiInputHandle,
    _make_unique_id,
)
from ._icons import svg_from_icon
from ._icons_enum import IconName
from ._message_api import MessageApi, cast_vector
from ._messages import FileTransferPartAck

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from .infra import ClientId

IntOrFloat = TypeVar("IntOrFloat", int, float)
TString = TypeVar("TString", bound=str)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)
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


@functools.lru_cache(maxsize=None)
def get_type_hints_cached(cls: Type[Any]) -> Dict[str, Any]:
    return get_type_hints(cls)  # type: ignore


def cast_value(tp, value):
    """Cast a value to a type, or raise a TypeError if it cannot be cast."""
    origin = get_origin(tp)

    if (origin is tuple or tp is tuple) and isinstance(value, list):
        return cast_value(tp, tuple(value))

    if origin is Literal:
        for val in get_args(tp):
            try:
                value_casted = cast_value(type(val), value)
                if val == value_casted:
                    return value_casted
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {get_args(tp)}")

    if origin is Union:
        for t in get_args(tp):
            try:
                return cast_value(t, value)
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {tp}")

    if tp in {int, float, bool, str}:
        return tp(value)

    if dataclasses.is_dataclass(tp):
        return tp(
            **{k: cast_value(v, value[k]) for k, v in get_type_hints_cached(tp).items()}
        )

    if isinstance(value, tp):
        return value

    raise TypeError(f"Cannot cast value {value} to type {tp}")


class FileUploadState(TypedDict):
    filename: str
    mime_type: str
    part_count: int
    parts: Dict[int, bytes]
    total_bytes: int
    transferred_bytes: int
    lock: threading.Lock


class GuiApi(abc.ABC):
    _target_container_from_thread_id: Dict[int, str] = {}
    """ID of container to put GUI elements into."""

    def __init__(self) -> None:
        super().__init__()

        self._gui_handle_from_id: Dict[str, _GuiInputHandle[Any]] = {}
        self._container_handle_from_id: Dict[str, GuiContainerProtocol] = {
            "root": _RootGuiContainer({})
        }
        self._current_file_upload_states: Dict[str, FileUploadState] = {}

        self._setup_plotly_js: bool = (
            False  # Set to True when plotly.min.js has been sent to client.
        )

        self._get_api()._message_handler.register_handler(
            _messages.GuiUpdateMessage, self._handle_gui_updates
        )
        self._get_api()._message_handler.register_handler(
            _messages.FileTransferStart, self._handle_file_transfer_start
        )
        self._get_api()._message_handler.register_handler(
            _messages.FileTransferPart,
            self._handle_file_transfer_part,
        )

    def _handle_gui_updates(
        self, client_id: ClientId, message: _messages.GuiUpdateMessage
    ) -> None:
        """Callback for handling GUI messages."""
        handle = self._gui_handle_from_id.get(message.id, None)
        if handle is None:
            return
        handle_state = handle._impl

        has_changed = False
        updates_cast = {}
        for prop_name, prop_value in message.updates.items():
            assert hasattr(handle_state, prop_name)
            current_value = getattr(handle_state, prop_name)

            # Do some type casting. This is brittle, but necessary when we
            # expect floats but the Javascript side gives us integers.
            if prop_name == "value":
                prop_value = cast_value(handle_state.typ, prop_value)

            # Update handle property.
            if current_value != prop_value:
                has_changed = True
                setattr(handle_state, prop_name, prop_value)

            # Save value, which might have been cast.
            updates_cast[prop_name] = prop_value

        # Only call update when value has actually changed.
        if not handle_state.is_button and not has_changed:
            return

        # GUI element has been updated!
        handle_state.update_timestamp = time.time()
        for cb in handle_state.update_cb:
            from ._viser import ClientHandle, ViserServer

            # Get the handle of the client that triggered this event.
            api = self._get_api()
            if isinstance(api, ClientHandle):
                client = api
            elif isinstance(api, ViserServer):
                client = api.get_clients()[client_id]
            else:
                assert False

            cb(GuiEvent(client, client_id, handle))

        if handle_state.sync_cb is not None:
            handle_state.sync_cb(client_id, updates_cast)

    def _handle_file_transfer_start(
        self, client_id: ClientId, message: _messages.FileTransferStart
    ) -> None:
        if message.source_component_id not in self._gui_handle_from_id:
            return
        self._current_file_upload_states[message.transfer_uuid] = {
            "filename": message.filename,
            "mime_type": message.mime_type,
            "part_count": message.part_count,
            "parts": {},
            "total_bytes": message.size_bytes,
            "transferred_bytes": 0,
            "lock": threading.Lock(),
        }

    def _handle_file_transfer_part(
        self, client_id: ClientId, message: _messages.FileTransferPart
    ) -> None:
        if message.transfer_uuid not in self._current_file_upload_states:
            return
        assert message.source_component_id in self._gui_handle_from_id

        state = self._current_file_upload_states[message.transfer_uuid]
        state["parts"][message.part] = message.content
        total_bytes = state["total_bytes"]

        with state["lock"]:
            state["transferred_bytes"] += len(message.content)

            # Send ack to the server.
            self._get_api()._queue(
                FileTransferPartAck(
                    source_component_id=message.source_component_id,
                    transfer_uuid=message.transfer_uuid,
                    transferred_bytes=state["transferred_bytes"],
                    total_bytes=total_bytes,
                )
            )

            if state["transferred_bytes"] < total_bytes:
                return

        # Finish the upload.
        assert state["transferred_bytes"] == total_bytes
        state = self._current_file_upload_states.pop(message.transfer_uuid)

        handle = self._gui_handle_from_id.get(message.source_component_id, None)
        if handle is None:
            return

        handle_state = handle._impl

        value = UploadedFile(
            name=state["filename"],
            content=b"".join(state["parts"][i] for i in range(state["part_count"])),
        )

        # Update state.
        with self._get_api()._atomic_lock:
            handle_state.value = value
            handle_state.update_timestamp = time.time()

        # Trigger callbacks.
        for cb in handle_state.update_cb:
            from ._viser import ClientHandle, ViserServer

            # Get the handle of the client that triggered this event.
            api = self._get_api()
            if isinstance(api, ClientHandle):
                client = api
            elif isinstance(api, ViserServer):
                client = api.get_clients()[client_id]
            else:
                assert False

            cb(GuiEvent(client, client_id, handle))

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

    if not TYPE_CHECKING:

        def gui_folder(self, label: str) -> GuiFolderHandle:
            """Deprecated."""
            warnings.warn(
                "gui_folder() is deprecated. Use add_gui_folder() instead!",
                stacklevel=2,
            )
            return self.add_gui_folder(label)

    def add_gui_folder(
        self,
        label: str,
        order: Optional[float] = None,
        expand_by_default: bool = True,
        visible: bool = True,
    ) -> GuiFolderHandle:
        """Add a folder, and return a handle that can be used to populate it.

        Args:
            label: Label to display on the folder.
            order: Optional ordering, smallest values will be displayed first.
            expand_by_default: Open the folder by default. Set to False to collapse it by
                default.
            visible: Whether the component is visible.

        Returns:
            A handle that can be used as a context to populate the folder.
        """
        folder_container_id = _make_unique_id()
        order = _apply_default_order(order)
        self._get_api()._queue(
            _messages.GuiAddFolderMessage(
                order=order,
                id=folder_container_id,
                label=label,
                container_id=self._get_container_id(),
                expand_by_default=expand_by_default,
                visible=visible,
            )
        )
        return GuiFolderHandle(
            _gui_api=self,
            _id=folder_container_id,
            _parent_container_id=self._get_container_id(),
            _order=order,
        )

    def add_gui_modal(
        self,
        title: str,
        order: Optional[float] = None,
    ) -> GuiModalHandle:
        """Show a modal window, which can be useful for popups and messages, then return
        a handle that can be used to populate it.

        Args:
            title: Title to display on the modal.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used as a context to populate the modal.
        """
        modal_container_id = _make_unique_id()
        order = _apply_default_order(order)
        self._get_api()._queue(
            _messages.GuiModalMessage(
                order=order,
                id=modal_container_id,
                title=title,
            )
        )
        return GuiModalHandle(
            _gui_api=self,
            _id=modal_container_id,
        )

    def add_gui_tab_group(
        self,
        order: Optional[float] = None,
        visible: bool = True,
    ) -> GuiTabGroupHandle:
        """Add a tab group.

        Args:
            order: Optional ordering, smallest values will be displayed first.
            visible: Whether the component is visible.

        Returns:
            A handle that can be used as a context to populate the tab group.
        """
        tab_group_id = _make_unique_id()
        order = _apply_default_order(order)

        self._get_api()._queue(
            _messages.GuiAddTabGroupMessage(
                order=order,
                id=tab_group_id,
                container_id=self._get_container_id(),
                tab_labels=(),
                visible=visible,
                tab_icons_html=(),
                tab_container_ids=(),
            )
        )
        return GuiTabGroupHandle(
            _tab_group_id=tab_group_id,
            _labels=[],
            _icons_html=[],
            _tabs=[],
            _gui_api=self,
            _order=order,
        )

    def add_gui_markdown(
        self,
        content: str,
        image_root: Optional[Path] = None,
        order: Optional[float] = None,
        visible: bool = True,
    ) -> GuiMarkdownHandle:
        """Add markdown to the GUI.

        Args:
            content: Markdown content to display.
            image_root: Optional root directory to resolve relative image paths.
            order: Optional ordering, smallest values will be displayed first.
            visible: Whether the component is visible.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        handle = GuiMarkdownHandle(
            _gui_api=self,
            _id=_make_unique_id(),
            _visible=visible,
            _container_id=self._get_container_id(),
            _order=_apply_default_order(order),
            _image_root=image_root,
            _content=None,
        )
        self._get_api()._queue(
            _messages.GuiAddMarkdownMessage(
                order=handle._order,
                id=handle._id,
                markdown="",
                container_id=handle._container_id,
                visible=visible,
            )
        )

        # Logic for processing markdown, handling images, etc is all in the
        # `.content` setter, which should send a GuiUpdateMessage.
        handle.content = content
        return handle

    def add_gui_plotly(
        self,
        figure: go.Figure,
        aspect_ratio: float = 1.0,
        order: Optional[float] = None,
        visible: bool = True,
    ) -> GuiPlotlyHandle:
        """Add a Plotly Plot to the GUI.

        Args:
            figure: Plotly figure to display.
            aspect_ratio: Aspect ratio of the plot in the control panel (width / height).
            order: Optional ordering, smallest values will be displayed first.
            visible: Whether the component is visible.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        handle = GuiPlotlyHandle(
            _gui_api=self,
            _id=_make_unique_id(),
            _visible=visible,
            _container_id=self._get_container_id(),
            _order=_apply_default_order(order),
            _figure=figure,
            _aspect_ratio=aspect_ratio,
        )

        # If plotly.min.js hasn't been sent to the client yet, the client won't be able
        # to render the plot. Send this large file now! (~3MB)
        if not self._setup_plotly_js:
            # Check if plotly is installed.
            try:
                import plotly
            except ImportError:
                raise ImportError(
                    "You must have the `plotly` package installed to use the Plotly GUI element."
                )

            # Check that plotly.min.js exists.
            plotly_path = (
                Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
            )
            assert (
                plotly_path.exists()
            ), f"Could not find plotly.min.js at {plotly_path}."

            # Send it over!
            with open(plotly_path, "r") as f:
                plotly_js = f.read()
            self._get_api()._queue(
                _messages.SetupPlotlyMessage(
                    source=plotly_js,
                )
            )

            # Update the flag so we don't send it again.
            self._setup_plotly_js = True

        # After plotly.min.js has been sent, we can send the plotly figure.
        self._get_api()._queue(
            _messages.GuiAddPlotlyMessage(
                order=handle._order,
                id=handle._id,
                plotly_json_str=handle.plot_to_json(),
                aspect_ratio=aspect_ratio,
                container_id=handle._container_id,
                visible=visible,
            )
        )
        return handle

    def add_gui_button(
        self,
        label: str,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        color: Optional[
            Literal[
                "dark",
                "gray",
                "red",
                "pink",
                "grape",
                "violet",
                "indigo",
                "blue",
                "cyan",
                "green",
                "lime",
                "yellow",
                "orange",
                "teal",
            ]
        ] = None,
        icon: Optional[IconName] = None,
        order: Optional[float] = None,
    ) -> GuiButtonHandle:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`.

        Args:
            label: Label to display on the button.
            visible: Whether the button is visible.
            disabled: Whether the button is disabled.
            hint: Optional hint to display on hover.
            color: Optional color to use for the button.
            icon: Optional icon to display on the button.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """

        # Re-wrap the GUI handle with a button interface.
        id = _make_unique_id()
        order = _apply_default_order(order)
        return GuiButtonHandle(
            self._create_gui_input(
                value=False,
                message=_messages.GuiAddButtonMessage(
                    order=order,
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    value=False,
                    color=color,
                    icon_html=None if icon is None else svg_from_icon(icon),
                    disabled=disabled,
                    visible=visible,
                ),
                is_button=True,
            )._impl
        )

    def add_gui_upload_button(
        self,
        label: str,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        color: Optional[
            Literal[
                "dark",
                "gray",
                "red",
                "pink",
                "grape",
                "violet",
                "indigo",
                "blue",
                "cyan",
                "green",
                "lime",
                "yellow",
                "orange",
                "teal",
            ]
        ] = None,
        icon: Optional[IconName] = None,
        mime_type: str = "*/*",
        order: Optional[float] = None,
    ) -> GuiUploadButtonHandle:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`.

        Args:
            label: Label to display on the button.
            visible: Whether the button is visible.
            disabled: Whether the button is disabled.
            hint: Optional hint to display on hover.
            color: Optional color to use for the button.
            icon: Optional icon to display on the button.
            mime_type: Optional MIME type to filter the files that can be uploaded.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """

        # Re-wrap the GUI handle with a button interface.
        id = _make_unique_id()
        order = _apply_default_order(order)
        return GuiUploadButtonHandle(
            self._create_gui_input(
                value=UploadedFile("", b""),
                message=_messages.GuiAddUploadButtonMessage(
                    value=None,
                    disabled=disabled,
                    visible=visible,
                    order=order,
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    color=color,
                    mime_type=mime_type,
                    icon_html=None if icon is None else svg_from_icon(icon),
                ),
                is_button=True,
            )._impl
        )

    # The TLiteralString overload tells pyright to resolve the value type to a Literal
    # whenever possible.
    #
    # TString is helpful when the input types are generic (could be str, could be
    # Literal).
    @overload
    def add_gui_button_group(
        self,
        label: str,
        options: Sequence[TLiteralString],
        visible: bool = True,
        disabled: bool = False,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiButtonGroupHandle[TLiteralString]:
        ...

    @overload
    def add_gui_button_group(
        self,
        label: str,
        options: Sequence[TString],
        visible: bool = True,
        disabled: bool = False,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiButtonGroupHandle[TString]:
        ...

    def add_gui_button_group(
        self,
        label: str,
        options: Sequence[TLiteralString] | Sequence[TString],
        visible: bool = True,
        disabled: bool = False,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiButtonGroupHandle[Any]:  # Return types are specified in overloads.
        """Add a button group to the GUI.

        Args:
            label: Label to display on the button group.
            options: Sequence of options to display as buttons.
            visible: Whether the button group is visible.
            disabled: Whether the button group is disabled.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = options[0]
        id = _make_unique_id()
        order = _apply_default_order(order)
        return GuiButtonGroupHandle(
            self._create_gui_input(
                value,
                message=_messages.GuiAddButtonGroupMessage(
                    order=order,
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    value=value,
                    options=tuple(options),
                    disabled=disabled,
                    visible=visible,
                ),
            )._impl,
        )

    def add_gui_checkbox(
        self,
        label: str,
        initial_value: bool,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[bool]:
        """Add a checkbox to the GUI.

        Args:
            label: Label to display on the checkbox.
            initial_value: Initial value of the checkbox.
            disabled: Whether the checkbox is disabled.
            visible: Whether the checkbox is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value
        assert isinstance(value, bool)
        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value,
            message=_messages.GuiAddCheckboxMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                disabled=disabled,
                visible=visible,
            ),
        )

    def add_gui_text(
        self,
        label: str,
        initial_value: str,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[str]:
        """Add a text input to the GUI.

        Args:
            label: Label to display on the text input.
            initial_value: Initial value of the text input.
            disabled: Whether the text input is disabled.
            visible: Whether the text input is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value
        assert isinstance(value, str)
        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value,
            message=_messages.GuiAddTextMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                disabled=disabled,
                visible=visible,
            ),
        )

    def add_gui_number(
        self,
        label: str,
        initial_value: IntOrFloat,
        min: Optional[IntOrFloat] = None,
        max: Optional[IntOrFloat] = None,
        step: Optional[IntOrFloat] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[IntOrFloat]:
        """Add a number input to the GUI, with user-specifiable bound and precision parameters.

        Args:
            label: Label to display on the number input.
            initial_value: Initial value of the number input.
            min: Optional minimum value of the number input.
            max: Optional maximum value of the number input.
            step: Optional step size of the number input. Computed automatically if not
                specified.
            disabled: Whether the number input is disabled.
            visible: Whether the number input is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value

        assert isinstance(value, (int, float))

        if step is None:
            # It's ok that `step` is always a float, even if the value is an integer,
            # because things all become `number` types after serialization.
            step = float(  # type: ignore
                onp.min(
                    [
                        _compute_step(value),
                        _compute_step(min),
                        _compute_step(max),
                    ]
                )
            )

        assert step is not None

        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value,
            message=_messages.GuiAddNumberMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                min=min,
                max=max,
                precision=_compute_precision_digits(step),
                step=step,
                disabled=disabled,
                visible=visible,
            ),
            is_button=False,
        )

    def add_gui_vector2(
        self,
        label: str,
        initial_value: Tuple[float, float] | onp.ndarray,
        min: Tuple[float, float] | onp.ndarray | None = None,
        max: Tuple[float, float] | onp.ndarray | None = None,
        step: Optional[float] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI.

        Args:
            label: Label to display on the vector input.
            initial_value: Initial value of the vector input.
            min: Optional minimum value of the vector input.
            max: Optional maximum value of the vector input.
            step: Optional step size of the vector input. Computed automatically if not
            disabled: Whether the vector input is disabled.
            visible: Whether the vector input is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value
        value = cast_vector(value, 2)
        min = cast_vector(min, 2) if min is not None else None
        max = cast_vector(max, 2) if max is not None else None
        id = _make_unique_id()
        order = _apply_default_order(order)

        if step is None:
            possible_steps: List[float] = []
            possible_steps.extend([_compute_step(x) for x in value])
            if min is not None:
                possible_steps.extend([_compute_step(x) for x in min])
            if max is not None:
                possible_steps.extend([_compute_step(x) for x in max])
            step = float(onp.min(possible_steps))

        return self._create_gui_input(
            value,
            message=_messages.GuiAddVector2Message(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                min=min,
                max=max,
                step=step,
                precision=_compute_precision_digits(step),
                disabled=disabled,
                visible=visible,
            ),
        )

    def add_gui_vector3(
        self,
        label: str,
        initial_value: Tuple[float, float, float] | onp.ndarray,
        min: Tuple[float, float, float] | onp.ndarray | None = None,
        max: Tuple[float, float, float] | onp.ndarray | None = None,
        step: Optional[float] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI.

        Args:
            label: Label to display on the vector input.
            initial_value: Initial value of the vector input.
            min: Optional minimum value of the vector input.
            max: Optional maximum value of the vector input.
            step: Optional step size of the vector input. Computed automatically if not
            disabled: Whether the vector input is disabled.
            visible: Whether the vector input is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value
        value = cast_vector(value, 2)
        min = cast_vector(min, 3) if min is not None else None
        max = cast_vector(max, 3) if max is not None else None
        id = _make_unique_id()
        order = _apply_default_order(order)

        if step is None:
            possible_steps: List[float] = []
            possible_steps.extend([_compute_step(x) for x in value])
            if min is not None:
                possible_steps.extend([_compute_step(x) for x in min])
            if max is not None:
                possible_steps.extend([_compute_step(x) for x in max])
            step = float(onp.min(possible_steps))

        return self._create_gui_input(
            value,
            message=_messages.GuiAddVector3Message(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                min=min,
                max=max,
                step=step,
                precision=_compute_precision_digits(step),
                disabled=disabled,
                visible=visible,
            ),
        )

    # See add_gui_dropdown for notes on overloads.
    @overload
    def add_gui_dropdown(
        self,
        label: str,
        options: Sequence[TLiteralString],
        initial_value: Optional[TLiteralString] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiDropdownHandle[TLiteralString]:
        ...

    @overload
    def add_gui_dropdown(
        self,
        label: str,
        options: Sequence[TString],
        initial_value: Optional[TString] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiDropdownHandle[TString]:
        ...

    def add_gui_dropdown(
        self,
        label: str,
        options: Sequence[TLiteralString] | Sequence[TString],
        initial_value: Optional[TLiteralString | TString] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiDropdownHandle[Any]:  # Output type is specified in overloads.
        """Add a dropdown to the GUI.

        Args:
            label: Label to display on the dropdown.
            options: Sequence of options to display in the dropdown.
            initial_value: Initial value of the dropdown.
            disabled: Whether the dropdown is disabled.
            visible: Whether the dropdown is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value
        if value is None:
            value = options[0]
        id = _make_unique_id()
        order = _apply_default_order(order)
        return GuiDropdownHandle(
            self._create_gui_input(
                value,
                message=_messages.GuiAddDropdownMessage(
                    order=order,
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    value=value,
                    options=tuple(options),
                    disabled=disabled,
                    visible=visible,
                ),
            )._impl,
            _impl_options=tuple(options),
        )

    def add_gui_slider(
        self,
        label: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: IntOrFloat,
        initial_value: IntOrFloat,
        marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[IntOrFloat]:
        """Add a slider to the GUI. Types of the min, max, step, and initial value should match.

        Args:
            label: Label to display on the slider.
            min: Minimum value of the slider.
            max: Maximum value of the slider.
            step: Step size of the slider.
            initial_value: Initial value of the slider.
            marks: Tuple of marks to display below the slider. Each mark should
                either be a numerical or a (number, label) tuple, where the
                label is provided as a string.
            disabled: Whether the slider is disabled.
            visible: Whether the slider is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value: IntOrFloat = initial_value
        assert max >= min
        if step > max - min:
            step = max - min
        assert max >= value >= min

        # GUI callbacks cast incoming values to match the type of the initial value. If
        # the min, max, or step is a float, we should cast to a float.
        #
        # This should also match what the IntOrFloat TypeVar resolves to.
        if type(value) is int and (
            type(min) is float or type(max) is float or type(step) is float
        ):
            value = float(value)  # type: ignore

        # TODO: as of 6/5/2023, this assert will break something in nerfstudio. (at
        # least LERF)
        #
        # assert type(min) == type(max) == type(step) == type(value)

        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value,
            message=_messages.GuiAddSliderMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                min=min,
                max=max,
                step=step,
                value=value,
                precision=_compute_precision_digits(step),
                visible=visible,
                disabled=disabled,
                marks=tuple(
                    {"value": float(x[0]), "label": x[1]}
                    if isinstance(x, tuple)
                    else {"value": float(x)}
                    for x in marks
                )
                if marks is not None
                else None,
            ),
            is_button=False,
        )

    def add_gui_multi_slider(
        self,
        label: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: IntOrFloat,
        initial_value: Tuple[IntOrFloat, ...],
        min_range: Optional[IntOrFloat] = None,
        fixed_endpoints: bool = False,
        marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[Tuple[IntOrFloat, ...]]:
        """Add a multi slider to the GUI. Types of the min, max, step, and initial value should match.

        Args:
            label: Label to display on the slider.
            min: Minimum value of the slider.
            max: Maximum value of the slider.
            step: Step size of the slider.
            initial_value: Initial values of the slider.
            min_range: Optional minimum difference between two values of the slider.
            fixed_endpoints: Whether the endpoints of the slider are fixed.
            marks: Tuple of marks to display below the slider. Each mark should
                either be a numerical or a (number, label) tuple, where the
                label is provided as a string.
            disabled: Whether the slider is disabled.
            visible: Whether the slider is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        assert max >= min
        if step > max - min:
            step = max - min
        assert all(max >= x >= min for x in initial_value)

        # GUI callbacks cast incoming values to match the type of the initial value. If
        # any of the arguments are floats, we should always use a float value.
        #
        # This should also match what the IntOrFloat TypeVar resolves to.
        if (
            type(min) is float
            or type(max) is float
            or type(step) is float
            or type(min_range) is float
        ):
            initial_value = tuple(float(x) for x in initial_value)  # type: ignore

        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value=initial_value,
            message=_messages.GuiAddMultiSliderMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                min=min,
                min_range=min_range,
                max=max,
                step=step,
                value=initial_value,
                visible=visible,
                disabled=disabled,
                fixed_endpoints=fixed_endpoints,
                precision=_compute_precision_digits(step),
                marks=tuple(
                    {"value": float(x[0]), "label": x[1]}
                    if isinstance(x, tuple)
                    else {"value": float(x)}
                    for x in marks
                )
                if marks is not None
                else None,
            ),
            is_button=False,
        )

    def add_gui_rgb(
        self,
        label: str,
        initial_value: Tuple[int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[Tuple[int, int, int]]:
        """Add an RGB picker to the GUI.

        Args:
            label: Label to display on the RGB picker.
            initial_value: Initial value of the RGB picker.
            disabled: Whether the RGB picker is disabled.
            visible: Whether the RGB picker is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """

        value = initial_value
        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value,
            message=_messages.GuiAddRgbMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                disabled=disabled,
                visible=visible,
            ),
        )

    def add_gui_rgba(
        self,
        label: str,
        initial_value: Tuple[int, int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
        order: Optional[float] = None,
    ) -> GuiInputHandle[Tuple[int, int, int, int]]:
        """Add an RGBA picker to the GUI.

        Args:
            label: Label to display on the RGBA picker.
            initial_value: Initial value of the RGBA picker.
            disabled: Whether the RGBA picker is disabled.
            visible: Whether the RGBA picker is visible.
            hint: Optional hint to display on hover.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        value = initial_value
        id = _make_unique_id()
        order = _apply_default_order(order)
        return self._create_gui_input(
            value,
            message=_messages.GuiAddRgbaMessage(
                order=order,
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                value=value,
                disabled=disabled,
                visible=visible,
            ),
        )

    def _create_gui_input(
        self,
        value: T,
        message: _messages._GuiAddInputBase,
        is_button: bool = False,
    ) -> GuiInputHandle[T]:
        """Private helper for adding a simple GUI element."""

        # Send add GUI input message.
        self._get_api()._queue(message)

        # Construct handle.
        handle_state = _GuiHandleState(
            label=message.label,
            message_type=type(message),
            typ=type(value),
            gui_api=self,
            value=value,
            update_timestamp=time.time(),
            container_id=self._get_container_id(),
            update_cb=[],
            is_button=is_button,
            sync_cb=None,
            disabled=message.disabled,
            visible=message.visible,
            id=message.id,
            order=message.order,
            hint=message.hint,
        )

        # For broadcasted GUI handles, we should synchronize all clients.
        # This will be a no-op for client handles.
        if not is_button:

            def sync_other_clients(
                client_id: ClientId, updates: Dict[str, Any]
            ) -> None:
                message = _messages.GuiUpdateMessage(handle_state.id, updates)
                message.excluded_self_client = client_id
                self._get_api()._queue(message)

            handle_state.sync_cb = sync_other_clients

        handle = GuiInputHandle(handle_state)

        return handle
