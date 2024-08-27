from __future__ import annotations

import colorsys
import dataclasses
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Tuple, TypeVar, cast, overload

import numpy as onp
from typing_extensions import (
    Literal,
    LiteralString,
    TypeAlias,
    TypedDict,
    get_type_hints,
)

from viser import theme

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
    GuiProgressBarHandle,
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
from ._messages import FileTransferPartAck
from ._scene_api import cast_vector

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ._viser import ClientHandle, ViserServer
    from .infra import ClientId


IntOrFloat = TypeVar("IntOrFloat", int, float)
TString = TypeVar("TString", bound=str)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)
T = TypeVar("T")
LengthTenStrTuple: TypeAlias = Tuple[str, str, str, str, str, str, str, str, str, str]
Color: TypeAlias = Literal[
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


def _hex_from_hls(h: float, l: float, s: float) -> str:
    """Converts HLS values in [0.0, 1.0] to a hex-formatted string, eg 0xffffff."""
    return "#" + "".join(
        [
            int(min(255, max(0, channel * 255.0)) + 0.5).to_bytes(1, "little").hex()
            for channel in colorsys.hls_to_rgb(h, l, s)
        ]
    )


def _compute_step(x: float | None) -> float:  # type: ignore
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
    _children: dict[str, SupportsRemoveProtocol]


_global_order_counter = 0


def _apply_default_order(order: float | None) -> float:
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
def get_type_hints_cached(cls: type[Any]) -> dict[str, Any]:
    return get_type_hints(cls)  # type: ignore


class _FileUploadState(TypedDict):
    filename: str
    mime_type: str
    part_count: int
    parts: dict[int, bytes]
    total_bytes: int
    transferred_bytes: int
    lock: threading.Lock


class GuiApi:
    """Interface for working with the 2D GUI in viser.

    Used by both our global server object, for sharing the same GUI elements
    with all clients, and by individual client handles."""

    _target_container_from_thread_id: dict[int, str] = {}
    """ID of container to put GUI elements into."""

    def __init__(
        self,
        owner: ViserServer | ClientHandle,  # Who do I belong to?
        thread_executor: ThreadPoolExecutor,
    ) -> None:
        from ._viser import ViserServer

        self._owner = owner
        """Entity that owns this API."""
        self._thread_executor = thread_executor

        self._websock_interface = (
            owner._websock_server
            if isinstance(owner, ViserServer)
            else owner._websock_connection
        )
        """Interface for sending and listening to messages."""

        self._gui_input_handle_from_id: dict[str, _GuiInputHandle[Any]] = {}
        self._container_handle_from_id: dict[str, GuiContainerProtocol] = {
            "root": _RootGuiContainer({})
        }
        self._current_file_upload_states: dict[str, _FileUploadState] = {}

        # Set to True when plotly.min.js has been sent to client.
        self._setup_plotly_js: bool = False

        self._websock_interface.register_handler(
            _messages.GuiUpdateMessage, self._handle_gui_updates
        )
        self._websock_interface.register_handler(
            _messages.FileTransferStart, self._handle_file_transfer_start
        )
        self._websock_interface.register_handler(
            _messages.FileTransferPart,
            self._handle_file_transfer_part,
        )

    def _handle_gui_updates(
        self, client_id: ClientId, message: _messages.GuiUpdateMessage
    ) -> None:
        """Callback for handling GUI messages."""
        handle = self._gui_input_handle_from_id.get(message.id, None)
        if handle is None:
            return
        handle_state = handle._impl

        has_changed = False
        updates_cast = {}
        for prop_name, prop_value in message.updates.items():
            assert hasattr(handle_state, prop_name)
            current_value = getattr(handle_state, prop_name)

            # Do some type casting. This is brittle, but necessary (1) when we
            # expect floats but the Javascript side gives us integers or (2)
            # when we expect tuples but the Javascript side gives us lists.
            if prop_name == "value":
                if isinstance(handle_state.value, tuple):
                    # We currently assume all tuple types have length >0, and
                    # contents are all the same type.
                    assert len(handle_state.value) > 0
                    typ = type(handle_state.value[0])
                    assert all([type(x) == typ for x in handle_state.value])
                    prop_value = tuple([typ(new) for new in prop_value])
                else:
                    prop_value = type(handle_state.value)(prop_value)

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
            if isinstance(self._owner, ClientHandle):
                client = self._owner
            elif isinstance(self._owner, ViserServer):
                client = self._owner.get_clients()[client_id]
            else:
                assert False

            cb(GuiEvent(client, client_id, handle))

        if handle_state.sync_cb is not None:
            handle_state.sync_cb(client_id, updates_cast)

    def _handle_file_transfer_start(
        self, client_id: ClientId, message: _messages.FileTransferStart
    ) -> None:
        if message.source_component_id not in self._gui_input_handle_from_id:
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
        assert message.source_component_id in self._gui_input_handle_from_id

        state = self._current_file_upload_states[message.transfer_uuid]
        state["parts"][message.part] = message.content
        total_bytes = state["total_bytes"]

        with state["lock"]:
            state["transferred_bytes"] += len(message.content)

            # Send ack to the server.
            self._websock_interface.queue_message(
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

        handle = self._gui_input_handle_from_id.get(message.source_component_id, None)
        if handle is None:
            return

        handle_state = handle._impl

        value = UploadedFile(
            name=state["filename"],
            content=b"".join(state["parts"][i] for i in range(state["part_count"])),
        )

        # Update state.
        with self._owner.atomic():
            handle_state.value = value
            handle_state.update_timestamp = time.time()

        # Trigger callbacks.
        for cb in handle_state.update_cb:
            from ._viser import ClientHandle, ViserServer

            # Get the handle of the client that triggered this event.
            if isinstance(self._owner, ClientHandle):
                client = self._owner
            elif isinstance(self._owner, ViserServer):
                client = self._owner.get_clients()[client_id]
            else:
                assert False

            cb(GuiEvent(client, client_id, handle))

    def _get_container_id(self) -> str:
        """Get container ID associated with the current thread."""
        return self._target_container_from_thread_id.get(threading.get_ident(), "root")

    def _set_container_id(self, container_id: str) -> None:
        """Set container ID associated with the current thread."""
        self._target_container_from_thread_id[threading.get_ident()] = container_id

    def reset(self) -> None:
        """Reset the GUI."""
        self._websock_interface.queue_message(_messages.ResetGuiMessage())

    def set_panel_label(self, label: str | None) -> None:
        """Set the main label that appears in the GUI panel.

        Args:
            label: The new label.
        """
        self._websock_interface.queue_message(_messages.SetGuiPanelLabelMessage(label))

    def configure_theme(
        self,
        *,
        titlebar_content: theme.TitlebarConfig | None = None,
        control_layout: Literal["floating", "collapsible", "fixed"] = "floating",
        control_width: Literal["small", "medium", "large"] = "medium",
        dark_mode: bool = False,
        show_logo: bool = True,
        show_share_button: bool = True,
        brand_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Configures the visual appearance of the viser front-end.

        Args:
            titlebar_content: Optional configuration for the title bar.
            control_layout: The layout of control elements, options are "floating",
                            "collapsible", or "fixed".
            control_width: The width of control elements, options are "small",
                           "medium", or "large".
            dark_mode: A boolean indicating if dark mode should be enabled.
            show_logo: A boolean indicating if the logo should be displayed.
            show_share_button: A boolean indicating if the share button should be displayed.
            brand_color: An optional tuple of integers (RGB) representing the brand color.
        """

        colors_cast: LengthTenStrTuple | None = None

        if brand_color is not None:
            assert len(brand_color) in (3, 10)
            if len(brand_color) == 3:
                assert all(
                    map(lambda val: isinstance(val, int), brand_color)
                ), "All channels should be integers."

                # RGB => HLS.
                h, l, s = colorsys.rgb_to_hls(
                    brand_color[0] / 255.0,
                    brand_color[1] / 255.0,
                    brand_color[2] / 255.0,
                )

                # Automatically generate a 10-color palette.
                min_l = max(l - 0.08, 0.0)
                max_l = min(0.8 + 0.5, 0.9)
                l = max(min_l, min(max_l, l))

                primary_index = 8
                ls = tuple(
                    onp.interp(
                        x=onp.arange(10),
                        xp=onp.array([0, primary_index, 9]),
                        fp=onp.array([max_l, l, min_l]),
                    )
                )
                colors_cast = cast(
                    LengthTenStrTuple,
                    tuple(_hex_from_hls(h, ls[i], s) for i in range(10)),
                )

        assert colors_cast is None or all(
            [isinstance(val, str) and val.startswith("#") for val in colors_cast]
        ), "All string colors should be in hexadecimal + prefixed with #, eg #ffffff."

        self._websock_interface.queue_message(
            _messages.ThemeConfigurationMessage(
                titlebar_content=titlebar_content,
                control_layout=control_layout,
                control_width=control_width,
                dark_mode=dark_mode,
                show_logo=show_logo,
                show_share_button=show_share_button,
                colors=colors_cast,
            ),
        )

    def add_folder(
        self,
        label: str,
        order: float | None = None,
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
        self._websock_interface.queue_message(
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

    def add_modal(
        self,
        title: str,
        order: float | None = None,
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
        self._websock_interface.queue_message(
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

    def add_tab_group(
        self,
        order: float | None = None,
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

        self._websock_interface.queue_message(
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
            _parent_container_id=self._get_container_id(),
            _order=order,
        )

    def add_markdown(
        self,
        content: str,
        image_root: Path | None = None,
        order: float | None = None,
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
            _parent_container_id=self._get_container_id(),
            _order=_apply_default_order(order),
            _image_root=image_root,
            _content=None,
        )
        self._websock_interface.queue_message(
            _messages.GuiAddMarkdownMessage(
                order=handle._order,
                id=handle._id,
                markdown="",
                container_id=handle._parent_container_id,
                visible=visible,
            )
        )

        # Logic for processing markdown, handling images, etc is all in the
        # `.content` setter, which should send a GuiUpdateMessage.
        handle.content = content
        return handle

    def add_plotly(
        self,
        figure: go.Figure,
        aspect: float = 1.0,
        order: float | None = None,
        visible: bool = True,
    ) -> GuiPlotlyHandle:
        """Add a Plotly figure to the GUI. Requires the `plotly` package to be
        installed.

        Args:
            figure: Plotly figure to display.
            aspect: Aspect ratio of the plot in the control panel (width/height).
            order: Optional ordering, smallest values will be displayed first.
            visible: Whether the component is visible.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        handle = GuiPlotlyHandle(
            _gui_api=self,
            _id=_make_unique_id(),
            _visible=visible,
            _parent_container_id=self._get_container_id(),
            _order=_apply_default_order(order),
            _figure=None,
            _aspect=None,
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
            plotly_js = plotly_path.read_text(encoding="utf-8")
            self._websock_interface.queue_message(
                _messages.RunJavascriptMessage(source=plotly_js)
            )

            # Update the flag so we don't send it again.
            self._setup_plotly_js = True

        # After plotly.min.js has been sent, we can send the plotly figure.
        # Empty string for `plotly_json_str` is a signal to the client to render nothing.
        self._websock_interface.queue_message(
            _messages.GuiAddPlotlyMessage(
                order=handle._order,
                id=handle._id,
                plotly_json_str="",
                aspect=1.0,
                container_id=handle._parent_container_id,
                visible=visible,
            )
        )

        # Set the plotly handle properties.
        handle.figure = figure
        handle.aspect = aspect

        return handle

    def add_button(
        self,
        label: str,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        color: Color | None = None,
        icon: IconName | None = None,
        order: float | None = None,
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

    def add_upload_button(
        self,
        label: str,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        color: Color | None = None,
        icon: IconName | None = None,
        mime_type: str = "*/*",
        order: float | None = None,
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
    def add_button_group(
        self,
        label: str,
        options: Sequence[TLiteralString],
        visible: bool = True,
        disabled: bool = False,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiButtonGroupHandle[TLiteralString]: ...

    @overload
    def add_button_group(
        self,
        label: str,
        options: Sequence[TString],
        visible: bool = True,
        disabled: bool = False,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiButtonGroupHandle[TString]: ...

    def add_button_group(
        self,
        label: str,
        options: Sequence[TLiteralString] | Sequence[TString],
        visible: bool = True,
        disabled: bool = False,
        hint: str | None = None,
        order: float | None = None,
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

    def add_checkbox(
        self,
        label: str,
        initial_value: bool,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
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

    def add_text(
        self,
        label: str,
        initial_value: str,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
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

    def add_number(
        self,
        label: str,
        initial_value: IntOrFloat,
        min: IntOrFloat | None = None,
        max: IntOrFloat | None = None,
        step: IntOrFloat | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
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

    def add_vector2(
        self,
        label: str,
        initial_value: tuple[float, float] | onp.ndarray,
        min: tuple[float, float] | onp.ndarray | None = None,
        max: tuple[float, float] | onp.ndarray | None = None,
        step: float | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiInputHandle[tuple[float, float]]:
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
            possible_steps: list[float] = []
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

    def add_vector3(
        self,
        label: str,
        initial_value: tuple[float, float, float] | onp.ndarray,
        min: tuple[float, float, float] | onp.ndarray | None = None,
        max: tuple[float, float, float] | onp.ndarray | None = None,
        step: float | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiInputHandle[tuple[float, float, float]]:
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
        value = cast_vector(value, 3)
        min = cast_vector(min, 3) if min is not None else None
        max = cast_vector(max, 3) if max is not None else None
        id = _make_unique_id()
        order = _apply_default_order(order)

        if step is None:
            possible_steps: list[float] = []
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

    # See add_dropdown for notes on overloads.
    @overload
    def add_dropdown(
        self,
        label: str,
        options: Sequence[TLiteralString],
        initial_value: TLiteralString | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiDropdownHandle[TLiteralString]: ...

    @overload
    def add_dropdown(
        self,
        label: str,
        options: Sequence[TString],
        initial_value: TString | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiDropdownHandle[TString]: ...

    def add_dropdown(
        self,
        label: str,
        options: Sequence[TLiteralString] | Sequence[TString],
        initial_value: TLiteralString | TString | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
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

    def add_progress_bar(
        self,
        value: float,
        visible: bool = True,
        animated: bool = False,
        color: Color | None = None,
        order: float | None = None,
    ) -> GuiProgressBarHandle:
        """Add a progress bar to the GUI.

        Args:
            value: Value of the progress bar. (0 - 100)
            visible: Whether the progress bar is visible.
            animated: Whether the progress bar is in a loading state (animated, striped).
            color: The color of the progress bar.
            order: Optional ordering, smallest values will be displayed first.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        assert value >= 0 and value <= 100
        handle = GuiProgressBarHandle(
            _gui_api=self,
            _id=_make_unique_id(),
            _visible=visible,
            _animated=animated,
            _parent_container_id=self._get_container_id(),
            _order=_apply_default_order(order),
            _value=value,
        )
        self._websock_interface.queue_message(
            _messages.GuiAddProgressBarMessage(
                order=handle._order,
                id=handle._id,
                value=value,
                animated=animated,
                color=color,
                container_id=handle._parent_container_id,
                visible=visible,
            )
        )
        return handle

    def add_slider(
        self,
        label: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: IntOrFloat,
        initial_value: IntOrFloat,
        marks: tuple[IntOrFloat | tuple[IntOrFloat, str], ...] | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiInputHandle[IntOrFloat]:
        """Add a slider to the GUI. Types of the min, max, step, and initial value should match.

        Args:
            label: Label to display on the slider.
            min: Minimum value of the slider.
            max: Maximum value of the slider.
            step: Step size of the slider.
            initial_value: Initial value of the slider.
            marks: tuple of marks to display below the slider. Each mark should
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
        step = __builtins__.min(step, max - min)
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
                    (
                        {"value": float(x[0]), "label": x[1]}
                        if isinstance(x, tuple)
                        else {"value": float(x)}
                    )
                    for x in marks
                )
                if marks is not None
                else None,
            ),
            is_button=False,
        )

    def add_multi_slider(
        self,
        label: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: IntOrFloat,
        initial_value: tuple[IntOrFloat, ...],
        min_range: IntOrFloat | None = None,
        fixed_endpoints: bool = False,
        marks: tuple[IntOrFloat | tuple[IntOrFloat, str], ...] | None = None,
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiInputHandle[tuple[IntOrFloat, ...]]:
        """Add a multi slider to the GUI. Types of the min, max, step, and initial value should match.

        Args:
            label: Label to display on the slider.
            min: Minimum value of the slider.
            max: Maximum value of the slider.
            step: Step size of the slider.
            initial_value: Initial values of the slider.
            min_range: Optional minimum difference between two values of the slider.
            fixed_endpoints: Whether the endpoints of the slider are fixed.
            marks: tuple of marks to display below the slider. Each mark should
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
        step = __builtins__.min(step, max - min)
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
                    (
                        {"value": float(x[0]), "label": x[1]}
                        if isinstance(x, tuple)
                        else {"value": float(x)}
                    )
                    for x in marks
                )
                if marks is not None
                else None,
            ),
            is_button=False,
        )

    def add_rgb(
        self,
        label: str,
        initial_value: tuple[int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiInputHandle[tuple[int, int, int]]:
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

    def add_rgba(
        self,
        label: str,
        initial_value: tuple[int, int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: str | None = None,
        order: float | None = None,
    ) -> GuiInputHandle[tuple[int, int, int, int]]:
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
        self._websock_interface.queue_message(message)

        # Construct handle.
        handle_state = _GuiHandleState(
            label=message.label,
            message_type=type(message),
            gui_api=self,
            value=value,
            update_timestamp=time.time(),
            parent_container_id=self._get_container_id(),
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
                client_id: ClientId, updates: dict[str, Any]
            ) -> None:
                message = _messages.GuiUpdateMessage(handle_state.id, updates)
                message.excluded_self_client = client_id
                self._websock_interface.queue_message(message)

            handle_state.sync_cb = sync_other_clients

        handle = GuiInputHandle(handle_state)

        return handle
