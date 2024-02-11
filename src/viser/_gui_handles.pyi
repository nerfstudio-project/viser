import typing

class Button:
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
    _: dataclasses.KW_ONLY
    color: Optional[Color]
    icon_base64: Optional[str]
    disabled: bool
    hint: Optional[str]
    def __init__(self, label: str, order: typing.Optional[float] = None, visible: bool = True, color: Optional[Color] = None, disabled: bool = False, hint: Optional[str] = None):
        """Add a button to the GUI. The value of this input is set to `True` every time
    it is clicked; to detect clicks, we can manually set it back to `False`.

    Args:
        label: Label to display on the button.
        visible: Whether the button is visible.
        disabled: Whether the button is disabled.
        hint: Optional hint to display on hover.
        icon: Optional icon to display on the button.
        order: Optional ordering, smallest values will be displayed first.


Args:
    label: The text of the button
    color: Optional color to use for the button.
    disabled: Whether the text input is disabled.
    hint: Optional hint to display on hover.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def __set_icon__(self, icon: Optional[IconName] = None) -> None:
        """Optional icon to display on the button."""
        ...

    def install(self, api: MessageApi) -> None:
        ...

    def on_click(self, callback: Callable[[GuiEvent['Button']], None]) -> None:
        ...


class ButtonGroup:
    """Handle for a button group input in our visualizer.

    Lets us detect clicks."""
    options: Sequence[str]
    def __init__(self, options: Sequence[str], label: str, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Handle for a button group input in our visualizer.

    Lets us detect clicks.

Args:
    options: Sequence of options to display as buttons.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...

    def on_click(self, func: Callable[[GuiEventWithPayload['ButtonGroup', str]], None]) -> Callable[[GuiEventWithPayload['ButtonGroup', str]], None]:
        """Attach a function to call when a button is pressed. Happens in a thread."""
        ...


class Checkbox:
    """Add a checkbox to the GUI."""
    value: bool
    def __init__(self, label: str, value: bool = False, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a checkbox to the GUI.

Args:
    value: Value of the checkbox.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Dropdown:
    """Handle for a dropdown-style GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""
    options: Tuple[str]
    value: str
    def __init__(self, options: Sequence[str], label: str, value: Optional[str] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Handle for a dropdown-style GUI input in our visualizer.

    Lets us get values, set values, and detect updates.

Args:
    options: Sequence of options to display in the dropdown.
    value: Value of the dropdown
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Folder:
    """Add a folder, and return a handle that can be used to populate it."""
    label: str
    _: dataclasses.KW_ONLY
    expand_by_default: bool
    def <lambda>(self):
        ...

    def <lambda>(self, args):
        ...

    def __init__(self, label: str, order: typing.Optional[float] = None, visible: bool = True, expand_by_default: bool = True):
        """Add a folder, and return a handle that can be used to populate it.

Args:
    label: Label to display on the folder.
    expand_by_default: Open the folder by default. Set to False to collapse it by default.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class GuiEvent:
    """Information associated with a GUI event, such as an update or click.

    Passed as input to callback functions."""
    client: Optional['_viser.ClientHandle']
    client_id: Optional[int]
    target: TGuiHandle
    def __delattr__(self, name):
        """Implement delattr(self, name)."""
        ...

    def __eq__(self, other):
        """Return self==value."""
        ...

    def __hash__(self):
        """Return hash(self)."""
        ...

    def __init__(self, client: Optional['_viser.ClientHandle'], client_id: Optional[int], target: TGuiHandle) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def __repr__(self):
        """Return repr(self)."""
        ...

    def __setattr__(self, name, value):
        """Implement setattr(self, name, value)."""
        ...


class GuiEventWithPayload:
    """Information associated with a GUI event, such as an update or click.

    Passed as input to callback functions."""
    client: Optional[ClientHandle]
    client_id: Optional[int]
    target: TGuiHandle
    payload: TPayload
    def __delattr__(self, name):
        """Implement delattr(self, name)."""
        ...

    def __eq__(self, other):
        """Return self==value."""
        ...

    def __hash__(self):
        """Return hash(self)."""
        ...

    def __init__(self, client: Optional[ClientHandle], client_id: Optional[int], target: TGuiHandle, payload: TPayload) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def __repr__(self):
        """Return repr(self)."""
        ...

    def __setattr__(self, name, value):
        """Implement setattr(self, name, value)."""
        ...


class GuiSliderMark:
    value: float
    label: typing_extensions.NotRequired[str]

class InputComponent:
    """Base class for all input components."""
    label: str
    _: dataclasses.KW_ONLY
    disabled: bool
    visible: bool
    hint: Optional[str]

class Markdown:
    """Add markdown to the GUI."""
    content: Optional[str]
    _: dataclasses.KW_ONLY
    image_root: Optional[Path]
    hint: Optional[str]
    def __init__(self, content: Optional[str] = None, order: typing.Optional[float] = None, visible: bool = True, image_root: Optional[Path] = None, hint: Optional[str] = None):
        """Add markdown to the GUI.

Args:
    content: Markdown content to display.
    image_root: Optional root directory to resolve relative image paths.
    hint: Optional hint to display on hover.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def __prepare_message__(self, props):
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Modal:
    """Show a modal window, which can be useful for popups and messages, then return
    a handle that can be used to populate it."""
    title: str
    def close(self) -> None:
        """Close this modal and permananently remove all contained GUI elements."""
        ...


class MultiSlider:
    """Add a multi slider to the GUI. Types of the min, max, step, and initial value should match."""
    value: IntOrFloat
    _: dataclasses.KW_ONLY
    min: IntOrFloat
    max: IntOrFloat
    step: IntOrFloat
    min_range: Optional[IntOrFloat]
    fixed_endpoints: bool
    marks: Optional[Tuple[_messages.Mark]]
    def __init__(self, value: IntOrFloat, label: str, min: IntOrFloat, max: IntOrFloat, step: IntOrFloat, min_range: Optional[IntOrFloat] = None, fixed_endpoints: bool = False, marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a multi slider to the GUI. Types of the min, max, step, and initial value should match.

Args:
    value: Value of the slider.
    min: Minimum value of the slider.
    max: Maximum value of the slider.
    step: Step size of the slider.
    min_range: Optional minimum difference between two values of the slider.
    fixed_endpoints: Whether the endpoints of the slider are fixed.
    marks: Tuple of marks to display below the slider. Each mark should
        either be a numerical or a (number, label) tuple, where the
        label is provided as a string.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Number:
    """Add a number input to the GUI, with user-specifiable bound and precision parameters."""
    value: IntOrFloat
    _: dataclasses.KW_ONLY
    min: Optional[IntOrFloat]
    max: Optional[IntOrFloat]
    step: Optional[IntOrFloat]
    def __init__(self, value: IntOrFloat, label: str, min: Optional[IntOrFloat] = None, max: Optional[IntOrFloat] = None, step: Optional[IntOrFloat] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a number input to the GUI, with user-specifiable bound and precision parameters.

Args:
    value: Value of the number input.
    min: Optional minimum value of the number input.
    max: Optional maximum value of the number input.
    step: Optional step size of the number input. Computed automatically if not specified.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Rgb:
    """Add an RGB picker to the GUI."""
    value: Tuple[int, int, int]
    def __init__(self, value: Tuple[int, int, int], label: str, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add an RGB picker to the GUI.

Args:
    value: Value of the RGBA picker.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Rgba:
    """Add an RGBA picker to the GUI."""
    value: Tuple[int, int, int, int]
    def __init__(self, value: Tuple[int, int, int, int], label: str, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add an RGBA picker to the GUI.

Args:
    value: Value of the RGBA picker.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Slider:
    """Add a slider to the GUI. Types of the min, max, step, and initial value should match."""
    value: IntOrFloat
    _: dataclasses.KW_ONLY
    min: IntOrFloat
    max: IntOrFloat
    step: IntOrFloat
    marks: Optional[Tuple[_messages.Mark]]
    def __init__(self, value: IntOrFloat, label: str, min: IntOrFloat, max: IntOrFloat, step: IntOrFloat, marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a slider to the GUI. Types of the min, max, step, and initial value should match.

Args:
    value: Value of the slider.
    min: Minimum value of the slider.
    max: Maximum value of the slider.
    step: Step size of the slider.
    marks: Tuple of marks to display below the slider. Each mark should
        either be a numerical or a (number, label) tuple, where the
        label is provided as a string.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def __post_init__(self) -> None:
        ...

    def install(self, api: MessageApi) -> None:
        ...


class TabGroup:
    id: str
    container_id: str
    order: float
    visible: bool
    def __init__(self, order: typing.Optional[float] = None, visible: bool = True):
        """Creates a new TabGroup instance

Args:
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def add_tab(self, label: str, icon: Optional[IconName] = None) -> TabGroupTab:
        """Add a tab. Returns a handle we can use to add GUI elements to it."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class TabGroupTab:
    """Handle for a tab in our visualizer.

    Lets us add GUI elements to it and remove it."""
    label: str
    icon_base64: Optional[str]
    def <lambda>(self):
        ...

    def <lambda>(self, args):
        ...

    def __init__(self, label: str, order: typing.Optional[float] = None, visible: bool = True):
        """Handle for a tab in our visualizer.

    Lets us add GUI elements to it and remove it.

Args:
    label: Label to display on the tab.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def __set_icon__(self, icon: Optional[IconName] = None) -> None:
        """The icon to set"""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Text:
    value: str
    def __init__(self, value: str = '', order: typing.Optional[float] = None, visible: bool = True):
        """Creates a new Text instance

Args:
    value: Initial value of the text input.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Vector2:
    """Add a length-2 vector input to the GUI."""
    value: Tuple[float, float]
    _: dataclasses.KW_ONLY
    min: Tuple[float, float] | None
    max: Tuple[float, float] | None
    step: Optional[float]
    def __init__(self, value: Tuple[float, float] | np.ndarray, label: str, min: Tuple[float, float] | np.ndarray | None = None, max: Tuple[float, float] | np.ndarray | None = None, step: Optional[float] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a length-2 vector input to the GUI.

Args:
    value: Initial value of the vector input.
    min: Optional minimum value of the vector input.
    max: Optional maximum value of the vector input.
    step: Optional step size of the vector input. Computed automatically if not present.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


class Vector3:
    """Add a length-3 vector input to the GUI."""
    value: Tuple[float, float, float]
    _: dataclasses.KW_ONLY
    min: Tuple[float, float, float] | None
    max: Tuple[float, float, float] | None
    step: Optional[float]
    def __init__(self, value: Tuple[float, float, float] | np.ndarray, label: str, min: Tuple[float, float, float] | np.ndarray | None = None, max: Tuple[float, float, float] | np.ndarray | None = None, step: Optional[float] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a length-3 vector input to the GUI.

Args:
    value: Initial value of the vector input.
    min: Optional minimum value of the vector input.
    max: Optional maximum value of the vector input.
    step: Optional step size of the vector input. Computed automatically if not present.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def install(self, api: MessageApi) -> None:
        ...


def _compute_precision_digits(x: float) -> int:
    """For number inputs: compute digits of precision from some number.

Example inputs/outputs:
    100 => 0
    12 => 0
    12.1 => 1
    10.2 => 1
    0.007 => 3"""
    ...

def _compute_step(x: Optional[float]) -> float:
    """For number inputs: compute an increment size from some number.

Example inputs/outputs:
    100 => 1
    12 => 1
    12.1 => 0.1
    12.02 => 0.01
    0.004 => 0.001"""
    ...

def _get_data_url(url: str, image_root: Optional[Path]) -> str:
    ...

def _parse_markdown(markdown: str, image_root: Optional[Path]) -> str:
    ...
