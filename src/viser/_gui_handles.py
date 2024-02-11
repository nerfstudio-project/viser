from __future__ import annotations

import dataclasses
import re
import urllib.parse
import warnings
from pathlib import Path
from typing import (
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
)

import imageio.v3 as iio
import numpy as np
from typing_extensions import NotRequired

from . import _messages
from ._gui_api_core import Component, ComponentMeta
from ._icons import base64_from_icon
from ._icons_enum import IconName
from ._message_api import _encode_image_base64, cast_vector


TGuiHandle = TypeVar("TGuiHandle")
TPayload = TypeVar("TPayload", bound=type)
IntOrFloat = TypeVar("IntOrFloat", int, float)
Color = Literal[
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
GuiSliderMark = TypedDict("GuiSliderMark", {"value": float, "label": NotRequired[str]})


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


@dataclasses.dataclass(frozen=True)
class GuiEvent(Generic[TGuiHandle]):
    """Information associated with a GUI event, such as an update or click.

    Passed as input to callback functions."""

    client: Optional["_viser.ClientHandle"]
    """Client that triggered this event."""
    client_id: Optional[int]
    """ID of client that triggered this event."""
    target: TGuiHandle
    """GUI element that was affected."""


@dataclasses.dataclass(frozen=True)
class GuiEventWithPayload(Generic[TGuiHandle, TPayload]):
    """Information associated with a GUI event, such as an update or click.

    Passed as input to callback functions."""

    client: Optional[ClientHandle]
    """Client that triggered this event."""
    client_id: Optional[int]
    """ID of client that triggered this event."""
    target: TGuiHandle
    """GUI element that was affected."""
    payload: TPayload
    """Payload associated with this event."""


class InputComponent(metaclass=ComponentMeta, abstract=True):
    """Base class for all input components."""

    label: str
    """Label to display on the text input."""
    _: dataclasses.KW_ONLY
    disabled: bool = False
    """Whether the text input is disabled."""
    visible: bool = True
    """Whether the text input is visible."""
    hint: Optional[str] = None
    """Optional hint to display on hover."""


class Button(Component, metaclass=ComponentMeta):
    """Add a button to the GUI. The value of this input is set to `True` every time
    it is clicked; to detect clicks, we can manually set it back to `False`.

    Args:
        label: Label to display on the button.
        visible: Whether the button is visible.
        disabled: Whether the button is disabled.
        hint: Optional hint to display on hover.
        icon: Optional icon to display on the button.
        order: Optional ordering, smallest values will be displayed first.
    """

    label: str
    """The text of the button"""

    _: dataclasses.KW_ONLY
    color: Optional[Color] = None
    """Optional color to use for the button."""
    icon_base64: Optional[str] = dataclasses.field(init=False, default=None)

    disabled: bool = False
    """Whether the text input is disabled."""
    hint: Optional[str] = None
    """Optional hint to display on hover."""

    def __set_icon__(self, icon: Optional[IconName] = None) -> None:
        """Optional icon to display on the button."""
        # Convert the icon to base64
        self.icon_base64 = base64_from_icon(icon)

    def on_click(self, callback: Callable[[GuiEvent["Button"]], None]) -> None:
        ...


class Text(Component, metaclass=ComponentMeta):
    value: str = ""
    """Initial value of the text input."""


class ButtonGroup(InputComponent, metaclass=ComponentMeta):
    """Handle for a button group input in our visualizer.

    Lets us detect clicks."""

    options: Sequence[str]
    """Sequence of options to display as buttons."""

    def on_click(
        self, func: Callable[[GuiEventWithPayload["ButtonGroup", str]], None]
    ) -> Callable[[GuiEventWithPayload["ButtonGroup", str]], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        ...


class Dropdown(InputComponent, metaclass=ComponentMeta):
    """Handle for a dropdown-style GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""

    options: Tuple[str]
    """Sequence of options to display in the dropdown."""
    value: str = dataclasses.field(kw_only=True, default=None)
    """Value of the dropdown"""

    def __set_value__(self, value: Optional[str] = None) -> None:
        """Value of the dropdown"""
        if value is None:
            value = self.options[0]
        self.value = value

    def __set_options__(self, options: Sequence[str]) -> None:
        """Sequence of options to display in the dropdown."""
        self.options = tuple(options)
        if self.value not in options:
            self.value = options[0]


class TabGroupTab(Component, metaclass=ComponentMeta, is_container=True):
    """Handle for a tab in our visualizer.

    Lets us add GUI elements to it and remove it."""

    label: str
    """Label to display on the tab."""
    icon_base64: Optional[str] = dataclasses.field(init=False, default=None)
    """Optional icon to display on the tab."""

    def __set_icon__(self, icon: Optional[IconName] = None) -> None:
        """The icon to set"""
        self.icon_base64 = base64_from_icon(icon)


class TabGroup(Component, metaclass=ComponentMeta):
    def add_tab(self, label: str, icon: Optional[IconName] = None) -> TabGroupTab:
        """Add a tab. Returns a handle we can use to add GUI elements to it."""
        return TabGroupTab(container_id=self.id, label=label, icon=icon)


class Folder(Component, metaclass=ComponentMeta, is_container=True):
    """Add a folder, and return a handle that can be used to populate it."""

    label: str
    """Label to display on the folder."""
    _: dataclasses.KW_ONLY
    expand_by_default: bool = True
    """Open the folder by default. Set to False to collapse it by default."""


class Modal:
    """Show a modal window, which can be useful for popups and messages, then return
    a handle that can be used to populate it."""

    title: str
    """Title to display on the modal."""

    def close(self) -> None:
        """Close this modal and permananently remove all contained GUI elements."""
        raise NotImplementedError()


def _get_data_url(url: str, image_root: Optional[Path]) -> str:
    if not url.startswith("http") and not image_root:
        warnings.warn(
            (
                "No `image_root` provided. All relative paths will be scoped to viser's"
                " installation path."
            ),
            stacklevel=2,
        )
    if url.startswith("http"):
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


def _parse_markdown(markdown: str, image_root: Optional[Path]) -> str:
    markdown = re.sub(
        r"\!\[([^]]*)\]\(([^]]*)\)",
        lambda match: (
            f"![{match.group(1)}]({_get_data_url(match.group(2), image_root)})"
        ),
        markdown,
    )
    return markdown


class Markdown(Component, metaclass=ComponentMeta):
    """Add markdown to the GUI."""

    content: Optional[str] = None
    """Markdown content to display."""
    _: dataclasses.KW_ONLY
    image_root: Optional[Path] = None
    """Optional root directory to resolve relative image paths."""
    hint: Optional[str] = None
    """Optional hint to display on hover."""

    def __prepare_message__(self, props):
        if "content" in props:
            content = props.pop("content")
            props["markdown"] = _parse_markdown(content, self.image_root)
        props.pop("image_root", None)
        return props


class Checkbox(InputComponent, metaclass=ComponentMeta):
    """Add a checkbox to the GUI."""

    value: bool = dataclasses.field(default=False, kw_only=True)
    """Value of the checkbox."""


class Number(InputComponent, metaclass=ComponentMeta):
    """Add a number input to the GUI, with user-specifiable bound and precision parameters."""

    value: IntOrFloat
    """Value of the number input."""
    _: dataclasses.KW_ONLY
    min: Optional[IntOrFloat] = None
    """Optional minimum value of the number input."""
    max: Optional[IntOrFloat] = None
    """Optional maximum value of the number input."""
    step: Optional[IntOrFloat]
    """Optional step size of the number input. Computed automatically if not specified."""

    def __set_step__(self, step: Optional[IntOrFloat] = None) -> None:
        if step is None:
            # It's ok that `step` is always a float, even if the value is an integer,
            # because things all become `number` types after serialization.
            step = float(  # type: ignore
                np.min(
                    [
                        _compute_step(self.value),
                        _compute_step(self.min),
                        _compute_step(self.max),
                    ]
                )
            )
        self.step = step


class Vector2(InputComponent, metaclass=ComponentMeta):
    """Add a length-2 vector input to the GUI."""

    value: Tuple[float, float]
    """Initial value of the vector input."""
    _: dataclasses.KW_ONLY
    min: Tuple[float, float] | None = None
    """Optional minimum value of the vector input."""
    max: Tuple[float, float] | None = None
    """Optional maximum value of the vector input."""
    step: Optional[float] = None
    """Optional step size of the vector input. Computed automatically if not present."""

    def __set_value__(self, value: Tuple[float, float] | np.ndarray) -> None:
        self.value = cast_vector(value, 2)

    def __set_min__(
        self, value: Tuple[float, float] | np.ndarray | None = None
    ) -> None:
        self.min = cast_vector(value, 2) if min is not None else None

    def __set_max__(
        self, value: Tuple[float, float] | np.ndarray | None = None
    ) -> None:
        self.max = cast_vector(value, 2) if max is not None else None

    def __set_step__(self, step: Optional[float] = None) -> None:
        # TODO: fix dependencies when setting values!!
        if step is None:
            possible_steps: List[float] = []
            possible_steps.extend([_compute_step(x) for x in self.value])
            if min is not None:
                possible_steps.extend([_compute_step(x) for x in self.min])
            if max is not None:
                possible_steps.extend([_compute_step(x) for x in self.max])
            step = float(np.min(possible_steps))
        self.step = step


class Vector3(InputComponent, metaclass=ComponentMeta):
    """Add a length-3 vector input to the GUI."""

    value: Tuple[float, float, float]
    """Initial value of the vector input."""
    _: dataclasses.KW_ONLY
    min: Tuple[float, float, float] | None = None
    """Optional minimum value of the vector input."""
    max: Tuple[float, float, float] | None = None
    """Optional maximum value of the vector input."""
    step: Optional[float] = None
    """Optional step size of the vector input. Computed automatically if not present."""

    def __set_value__(self, value: Tuple[float, float, float] | np.ndarray) -> None:
        self.value = cast_vector(value, 3)

    def __set_min__(
        self, value: Tuple[float, float, float] | np.ndarray | None = None
    ) -> None:
        self.min = cast_vector(value, 3) if min is not None else None

    def __set_max__(
        self, value: Tuple[float, float, float] | np.ndarray | None = None
    ) -> None:
        self.max = cast_vector(value, 3) if max is not None else None

    def __set_step__(self, step: Optional[float] = None) -> None:
        # TODO: fix dependencies when setting values!!
        if step is None:
            possible_steps: List[float] = []
            possible_steps.extend([_compute_step(x) for x in self.value])
            if min is not None:
                possible_steps.extend([_compute_step(x) for x in self.min])
            if max is not None:
                possible_steps.extend([_compute_step(x) for x in self.max])
            step = float(np.min(possible_steps))
        self.step = step


class Slider(InputComponent, metaclass=ComponentMeta):
    """Add a slider to the GUI. Types of the min, max, step, and initial value should match."""

    value: IntOrFloat
    """Value of the slider."""
    _: dataclasses.KW_ONLY
    min: IntOrFloat
    """Minimum value of the slider."""
    max: IntOrFloat
    """Maximum value of the slider."""
    step: IntOrFloat
    """Step size of the slider."""

    marks: Optional[Tuple[_messages.Mark]] = None

    def __set_marks__(
        self, marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None
    ) -> None:
        """Tuple of marks to display below the slider. Each mark should
        either be a numerical or a (number, label) tuple, where the
        label is provided as a string."""
        self.marks = (
            tuple(
                {"value": float(x[0]), "label": x[1]}
                if isinstance(x, tuple)
                else {"value": float(x)}
                for x in marks
            )
            if marks is not None
            else None
        )

    def __post_init__(self) -> None:
        assert self.max >= self.min
        if self.step > self.max - self.min:
            self.step = self.max - self.min
        assert self.max >= self.value >= self.min


class MultiSlider(InputComponent, metaclass=ComponentMeta):
    """Add a multi slider to the GUI. Types of the min, max, step, and initial value should match."""

    value: IntOrFloat
    """Value of the slider."""
    _: dataclasses.KW_ONLY
    min: IntOrFloat
    """Minimum value of the slider."""
    max: IntOrFloat
    """Maximum value of the slider."""
    step: IntOrFloat
    """Step size of the slider."""

    min_range: Optional[IntOrFloat] = None
    """Optional minimum difference between two values of the slider."""

    fixed_endpoints: bool = False
    """Whether the endpoints of the slider are fixed."""

    marks: Optional[Tuple[_messages.Mark]] = None

    def __set_marks__(
        self, marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None
    ) -> None:
        """Tuple of marks to display below the slider. Each mark should
        either be a numerical or a (number, label) tuple, where the
        label is provided as a string."""
        self.marks = (
            tuple(
                {"value": float(x[0]), "label": x[1]}
                if isinstance(x, tuple)
                else {"value": float(x)}
                for x in marks
            )
            if marks is not None
            else None
        )


class Rgb(InputComponent, metaclass=ComponentMeta):
    """Add an RGB picker to the GUI."""

    value: Tuple[int, int, int]
    """Value of the RGBA picker."""


class Rgba(InputComponent, metaclass=ComponentMeta):
    """Add an RGBA picker to the GUI."""

    value: Tuple[int, int, int, int]
    """Value of the RGBA picker."""
