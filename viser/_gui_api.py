from __future__ import annotations

import abc
import dataclasses
import time
import uuid
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import numpy as onp
from typing_extensions import LiteralString

from . import _messages
from ._gui_handles import (
    GuiButtonGroupHandle,
    GuiButtonHandle,
    GuiDropdownHandle,
    GuiHandle,
    _GuiHandleState,
)
from ._icons import base64_from_icon
from ._icons_enum import Icon
from ._message_api import MessageApi, cast_vector

if TYPE_CHECKING:
    from .infra import ClientId

IntOrFloat = TypeVar("IntOrFloat", int, float)
TString = TypeVar("TString", bound=str)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)
T = TypeVar("T")


def _make_unique_id() -> str:
    """Return a unique ID for referencing GUI elements."""
    return str(uuid.uuid4())


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


# Global override for placing GUI elements in containers.
container_id_override: Optional[str] = None

ROOT_CONTAINER_ID = "root"


class GuiApi(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def _get_api(self) -> MessageApi:
        """Message API to use."""
        ...

    def _get_container_id(self) -> str:
        """ID of container to put GUI elements into."""
        if container_id_override is not None:
            return container_id_override
        return ROOT_CONTAINER_ID

    def gui_folder(self, label: str) -> GuiFolderHandle:
        """Deprecated."""
        warnings.warn(
            "gui_folder() is deprecated. Use add_gui_folder() instead!",
            stacklevel=2,
        )
        return self.add_gui_folder(label)

    def add_gui_folder(self, label: str) -> GuiFolderHandle:
        """Add a folder, and return a handle that can be used to populate it."""
        folder_container_id = _make_unique_id()
        self._get_api()._queue(
            _messages.GuiAddFolderMessage(
                order=time.time(),
                id=folder_container_id,
                label=label,
                container_id=self._get_container_id(),
            )
        )
        return GuiFolderHandle(
            _api=self._get_api(),
            _folder_id=folder_container_id,
        )

    def add_gui_tab_group(self) -> GuiTabGroupHandle:
        """Add a tab group."""
        tab_group_id = _make_unique_id()
        return GuiTabGroupHandle(
            _tab_group_id=tab_group_id,
            _labels=[],
            _icons_base64=[],
            _tab_container_ids=[],
            _api=self._get_api(),
            _container_id=self._get_container_id(),
        )

    def add_gui_button(
        self,
        label: str,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiButtonHandle:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`."""

        # Re-wrap the GUI handle with a button interface.
        id = _make_unique_id()
        return GuiButtonHandle(
            self._create_gui_input(
                initial_value=False,
                message=_messages.GuiAddButtonMessage(
                    order=time.time(),
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    initial_value=False,
                ),
                disabled=disabled,
                visible=visible,
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
    ) -> GuiButtonGroupHandle[TString]:
        ...

    def add_gui_button_group(
        self,
        label: str,
        options: Sequence[TLiteralString] | Sequence[TString],
        visible: bool = True,
        disabled: bool = False,
        hint: Optional[str] = None,
    ) -> GuiButtonGroupHandle[Any]:  # Return types are specified in overloads.
        """Add a button group to the GUI."""
        initial_value = options[0]
        id = _make_unique_id()
        return GuiButtonGroupHandle(
            self._create_gui_input(
                initial_value,
                message=_messages.GuiAddButtonGroupMessage(
                    order=time.time(),
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    initial_value=initial_value,
                    options=tuple(options),
                ),
                disabled=disabled,
                visible=visible,
            )._impl,
        )

    def add_gui_checkbox(
        self,
        label: str,
        initial_value: bool,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[bool]:
        """Add a checkbox to the GUI."""
        assert isinstance(initial_value, bool)
        id = _make_unique_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddCheckboxMessage(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
            ),
            disabled=disabled,
            visible=visible,
        )

    def add_gui_text(
        self,
        label: str,
        initial_value: str,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[str]:
        """Add a text input to the GUI."""
        assert isinstance(initial_value, str)
        id = _make_unique_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddTextMessage(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
            ),
            disabled=disabled,
            visible=visible,
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
    ) -> GuiHandle[IntOrFloat]:
        """Add a number input to the GUI, with user-specifiable bound and precision parameters."""
        assert isinstance(initial_value, (int, float))

        if step is None:
            # It's ok that `step` is always a float, even if the value is an integer,
            # because things all become `number` types after serialization.
            step = float(  # type: ignore
                onp.min(
                    [
                        _compute_step(initial_value),
                        _compute_step(min),
                        _compute_step(max),
                    ]
                )
            )

        assert step is not None

        id = _make_unique_id()
        return self._create_gui_input(
            initial_value=initial_value,
            message=_messages.GuiAddNumberMessage(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
                min=min,
                max=max,
                precision=_compute_precision_digits(step),
                step=step,
            ),
            disabled=disabled,
            visible=visible,
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
    ) -> GuiHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI."""
        initial_value = cast_vector(initial_value, 2)
        min = cast_vector(min, 2) if min is not None else None
        max = cast_vector(max, 2) if max is not None else None
        id = _make_unique_id()

        if step is None:
            possible_steps = []
            possible_steps.extend([_compute_step(x) for x in initial_value])
            if min is not None:
                possible_steps.extend([_compute_step(x) for x in min])
            if max is not None:
                possible_steps.extend([_compute_step(x) for x in max])
            step = float(onp.min(possible_steps))

        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddVector2Message(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
                min=min,
                max=max,
                step=step,
                precision=_compute_precision_digits(step),
            ),
            disabled=disabled,
            visible=visible,
        )

    def add_gui_vector3(
        self,
        label: str,
        initial_value: Tuple[float, float, float] | onp.ndarray,
        min: Tuple[float, float, float] | onp.ndarray | None = None,
        max: Tuple[float, float, float] | onp.ndarray | None = None,
        step: Optional[float] = None,
        lock: bool = False,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI."""
        initial_value = cast_vector(initial_value, 2)
        min = cast_vector(min, 3) if min is not None else None
        max = cast_vector(max, 3) if max is not None else None
        id = _make_unique_id()

        if step is None:
            possible_steps = []
            possible_steps.extend([_compute_step(x) for x in initial_value])
            if min is not None:
                possible_steps.extend([_compute_step(x) for x in min])
            if max is not None:
                possible_steps.extend([_compute_step(x) for x in max])
            step = float(onp.min(possible_steps))

        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddVector3Message(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
                min=min,
                max=max,
                step=step,
                precision=_compute_precision_digits(step),
            ),
            disabled=disabled,
            visible=visible,
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
    ) -> GuiDropdownHandle[Any]:  # Output type is specified in overloads.
        """Add a dropdown to the GUI."""
        if initial_value is None:
            initial_value = options[0]
        id = _make_unique_id()
        return GuiDropdownHandle(
            self._create_gui_input(
                initial_value,
                message=_messages.GuiAddDropdownMessage(
                    order=time.time(),
                    id=id,
                    label=label,
                    container_id=self._get_container_id(),
                    hint=hint,
                    initial_value=initial_value,
                    options=tuple(options),
                ),
                disabled=disabled,
                visible=visible,
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
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[IntOrFloat]:
        """Add a slider to the GUI. Types of the min, max, step, and initial value should match."""
        assert max >= min
        if step > max - min:
            step = max - min
        assert max >= initial_value >= min

        # GUI callbacks cast incoming values to match the type of the initial value. If
        # the min, max, or step is a float, we should cast to a float.
        if type(initial_value) is int and (
            type(min) is float or type(max) is float or type(step) is float
        ):
            initial_value = float(initial_value)  # type: ignore

        # TODO: as of 6/5/2023, this assert will break something in nerfstudio. (at
        # least LERF)
        #
        # assert type(min) == type(max) == type(step) == type(initial_value)

        # Re-wrap the GUI handle with a button interface.
        id = _make_unique_id()
        return self._create_gui_input(
            initial_value=initial_value,
            message=_messages.GuiAddSliderMessage(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                min=min,
                max=max,
                step=step,
                initial_value=initial_value,
                precision=_compute_precision_digits(step),
            ),
            disabled=disabled,
            visible=visible,
            is_button=False,
        )

    def add_gui_rgb(
        self,
        label: str,
        initial_value: Tuple[int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[int, int, int]]:
        """Add an RGB picker to the GUI."""
        id = _make_unique_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddRgbMessage(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
            ),
            disabled=disabled,
            visible=visible,
        )

    def add_gui_rgba(
        self,
        label: str,
        initial_value: Tuple[int, int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[int, int, int, int]]:
        """Add an RGBA picker to the GUI."""
        id = _make_unique_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddRgbaMessage(
                order=time.time(),
                id=id,
                label=label,
                container_id=self._get_container_id(),
                hint=hint,
                initial_value=initial_value,
            ),
            disabled=disabled,
            visible=visible,
        )

    def _create_gui_input(
        self,
        initial_value: T,
        message: _messages._GuiAddInputBase,
        disabled: bool,
        visible: bool,
        is_button: bool = False,
    ) -> GuiHandle[T]:
        """Private helper for adding a simple GUI element."""

        # Send add GUI input message.
        self._get_api()._queue(message)

        # Construct handle.
        handle_state = _GuiHandleState(
            label=message.label,
            typ=type(initial_value),
            container=self,
            value=initial_value,
            update_timestamp=time.time(),
            container_id=self._get_container_id(),
            update_cb=[],
            is_button=is_button,
            sync_cb=None,
            cleanup_cb=None,
            disabled=False,
            visible=True,
            id=message.id,
            order=message.order,
            initial_value=initial_value,
            hint=message.hint,
        )
        self._get_api()._gui_handle_state_from_id[handle_state.id] = handle_state
        handle_state.cleanup_cb = lambda: self._get_api()._gui_handle_state_from_id.pop(
            handle_state.id
        )

        # For broadcasted GUI handles, we should synchronize all clients.
        # This will be a no-op for client handles.
        if not is_button:

            def sync_other_clients(client_id: ClientId, value: Any) -> None:
                message = _messages.GuiSetValueMessage(id=handle_state.id, value=value)
                message.excluded_self_client = client_id
                self._get_api()._queue(message)

            handle_state.sync_cb = sync_other_clients

        handle = GuiHandle(handle_state)

        # Set the disabled/visible fields. These will queue messages under-the-hood.
        if disabled:
            handle.disabled = disabled
        if not visible:
            handle.visible = visible

        return handle


@dataclasses.dataclass(frozen=True)
class GuiTabGroupHandle:
    _tab_group_id: str
    _labels: List[str]
    _icons_base64: List[Optional[str]]
    _tab_container_ids: List[str]
    _api: MessageApi
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
        self._api._queue(_messages.GuiRemoveMessage(self._tab_group_id))
        for tab_container_id in self._tab_container_ids:
            self._api._queue(
                _messages.GuiRemoveContainerChildrenMessage(tab_container_id)
            )

    def _sync_with_client(self) -> None:
        """Send a message that syncs tab state with the client."""
        self._api._queue(
            _messages.GuiAddTabGroupMessage(
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

    _api: MessageApi
    _folder_id: str
    _container_id_restore: Optional[str] = None

    def __enter__(self) -> None:
        # For folders, we use the same ID for the container ID / folder ID.
        global container_id_override
        self._container_id_restore = container_id_override
        container_id_override = self._folder_id

    def __exit__(self, *args) -> None:
        del args
        global container_id_override
        container_id_override = self._container_id_restore

    def remove(self) -> None:
        """Permanently remove this folder and all contained GUI elements from the
        visualizer."""
        self._api._queue(_messages.GuiRemoveMessage(self._folder_id))
        # Containers corresponding to folders are removed automatically.
        #
        # self._parent._api._queue(
        #     _messages.GuiRemoveContainerChildrenMessage(self._folder_id)
        # )


@dataclasses.dataclass
class GuiTabHandle:
    """Use as a context to place GUI elements into a tab."""

    _parent: GuiTabGroupHandle
    _container_id: str
    _container_id_restore: Optional[str] = None

    def __enter__(self) -> None:
        global container_id_override
        self._container_id_restore = container_id_override
        container_id_override = self._container_id

    def __exit__(self, *args) -> None:
        del args
        global container_id_override
        container_id_override = self._container_id_restore

    def remove(self) -> None:
        """Permanently remove this tab and all contained GUI elements from the
        visualizer."""
        # We may want to make this thread-safe in the future.
        container_index = self._parent._tab_container_ids.index(self._container_id)
        assert container_index != -1, "Tab already removed!"

        print(container_index)

        self._parent._labels.pop(container_index)
        self._parent._icons_base64.pop(container_index)
        self._parent._tab_container_ids.pop(container_index)

        self._parent._sync_with_client()
