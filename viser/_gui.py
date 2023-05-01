from __future__ import annotations

import dataclasses
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import numpy as onp

from ._messages import (
    GuiRemoveMessage,
    GuiSetLevaConfMessage,
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

    name: str
    typ: Type[T]
    api: MessageApi
    value: T
    update_timestamp: float

    folder_labels: List[str]
    """Name of the folders this GUI input was placed into."""

    update_cb: List[Callable[[Any], None]]
    """Registered functions to call when this input is updated."""

    leva_conf: Dict[str, Any]
    """Input config for Leva."""

    is_button: bool
    """Indicates a button element, which requires special handling."""

    sync_cb: Optional[Callable[[ClientId, T], None]]
    """Callback for synchronizing inputs across clients."""

    cleanup_cb: Optional[Callable[[], Any]]
    """Function to call when GUI element is removed."""

    # Encoder: run on outgoing message values.
    # Decoder: run on incoming message values.
    #
    # This helps us handle cases where types used by Leva don't match what we want to
    # expose as a Python API.
    encoder: Callable[[T], Any]
    decoder: Callable[[Any], T]

    disabled: bool
    visible: bool


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
                GuiSetValueMessage(self._impl.name, self._impl.encoder(value))  # type: ignore
            )

        # Set internal state. We automatically convert numpy arrays to the expected
        # internal type. (eg 1D arrays to tuples)
        self._impl.value = type(self._impl.value)(value)  # type: ignore
        self._impl.update_timestamp = time.time()

        # Call update callbacks.
        for cb in self._impl.update_cb:
            cb(self)

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

        if self._impl.is_button:
            self._impl.leva_conf["settings"]["disabled"] = disabled
            self._impl.api._queue(
                GuiSetLevaConfMessage(self._impl.name, self._impl.leva_conf),
            )
        else:
            self._impl.leva_conf["disabled"] = disabled
            self._impl.api._queue(
                GuiSetLevaConfMessage(self._impl.name, self._impl.leva_conf),
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

        self._impl.api._queue(GuiSetVisibleMessage(self._impl.name, visible=visible))
        self._impl.visible = visible

    def remove(self) -> None:
        """Permanently remove this GUI element from the visualizer."""
        self._impl.api._queue(GuiRemoveMessage(self._impl.name))
        assert self._impl.cleanup_cb is not None
        self._impl.cleanup_cb()


StringType = TypeVar("StringType", bound=str)


@dataclasses.dataclass
class GuiHandle(_GuiHandle[T], Generic[T]):
    """Handle for a particular GUI input in our visualizer.

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
class GuiSelectHandle(GuiHandle[StringType], Generic[StringType]):
    """Handle for a dropdown-style GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""

    _impl_options: List[StringType]

    @property
    def options(self) -> List[StringType]:
        """Options for our dropdown. Synchronized automatically when assigned.

        For projects that care about typing: the static type of `options` should be
        consistent with the `StringType` associated with a handle. Literal types will be
        inferred where possible when handles are instantiated; for the most flexibility,
        we can declare handles as `_GuiHandle[str]`.
        """
        return self._impl_options

    @options.setter
    def options(self, options: List[StringType]) -> None:
        self._impl_options = options

        # Make sure initial value is in options.
        self._impl.leva_conf["options"] = options
        if self._impl.leva_conf["value"] not in options:
            self._impl.leva_conf["value"] = options[0]

        # Update options.
        self._impl.api._queue(
            GuiSetLevaConfMessage(self._impl.name, self._impl.leva_conf),
        )

        # Make sure current value is in options.
        if self.value not in options:
            self.value = options[0]
