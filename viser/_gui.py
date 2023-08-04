from __future__ import annotations

import dataclasses
import threading
import time
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
    Literal,
)

import numpy as onp

from ._messages import (
    GuiAddDropdownMessage,
    GuiRemoveMessage,
    GuiSetDisabledMessage,
    GuiSetValueMessage,
    GuiSetVisibleMessage,
)
from .infra import ClientId

if TYPE_CHECKING:
    from ._message_api import MessageApi


T = TypeVar("T")
TGuiHandle = TypeVar("TGuiHandle", bound="_GuiHandle")


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    label: str
    typ: Type[T]
    api: MessageApi
    value: T
    update_timestamp: float

    folder_labels: Tuple[str, ...]
    """Name of the folders this GUI input was placed into."""

    destination: Literal["CONTROL_PANEL", "MODAL"]
    """Desination this GUI input was placed into."""

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
            self._impl.api._queue(
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

        self._impl.api._queue(GuiSetDisabledMessage(self._impl.id, disabled=disabled))
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

        self._impl.api._queue(GuiSetVisibleMessage(self._impl.id, visible=visible))
        self._impl.visible = visible

    def remove(self) -> None:
        """Permanently remove this GUI element from the visualizer."""
        self._impl.api._queue(GuiRemoveMessage(self._impl.id))
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

        self._impl.api._queue(
            GuiAddDropdownMessage(
                order=self._impl.order,
                id=self._impl.id,
                label=self._impl.label,
                folder_labels=self._impl.folder_labels,
                hint=self._impl.hint,
                initial_value=self._impl.initial_value,
                options=self._impl_options,
            )
        )

        if self.value not in self._impl_options:
            self.value = self._impl_options[0]
