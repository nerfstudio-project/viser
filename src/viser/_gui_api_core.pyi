import typing

class Component:
    id: str
    container_id: str
    order: float
    visible: bool

class ComponentMeta:
    def __init__(cls, name, bases, namespace, abstract: bool = False, is_container: bool = False):
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def __new__(cls, name, bases, namespace, abstract: bool = False, is_container: bool = False):
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

    def _build_add_component_method(cls, name):
        ...


class GuiApiMixin:
    def __init__(self, api: MessageApi) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def add_gui_button(self, label: str, order: typing.Optional[float] = None, visible: bool = True, color: Optional[Color] = None, disabled: bool = False, hint: Optional[str] = None):
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

    def add_gui_button_group(self, options: Sequence[str], label: str, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Handle for a button group input in our visualizer.

    Lets us detect clicks.

Args:
    options: Sequence of options to display as buttons.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def add_gui_checkbox(self, label: str, value: bool = False, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add a checkbox to the GUI.

Args:
    value: Value of the checkbox.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def add_gui_component(self, args, kwargs):
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def add_gui_dropdown(self, options: Sequence[str], label: str, value: Optional[str] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
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

    def add_gui_folder(self, label: str, order: typing.Optional[float] = None, visible: bool = True, expand_by_default: bool = True):
        """Add a folder, and return a handle that can be used to populate it.

Args:
    label: Label to display on the folder.
    expand_by_default: Open the folder by default. Set to False to collapse it by default.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def add_gui_input_component(self, args, kwargs):
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def add_gui_markdown(self, content: Optional[str] = None, order: typing.Optional[float] = None, visible: bool = True, image_root: Optional[Path] = None, hint: Optional[str] = None):
        """Add markdown to the GUI.

Args:
    content: Markdown content to display.
    image_root: Optional root directory to resolve relative image paths.
    hint: Optional hint to display on hover.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def add_gui_multi_slider(self, value: IntOrFloat, label: str, min: IntOrFloat, max: IntOrFloat, step: IntOrFloat, min_range: Optional[IntOrFloat] = None, fixed_endpoints: bool = False, marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
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

    def add_gui_number(self, value: IntOrFloat, label: str, min: Optional[IntOrFloat] = None, max: Optional[IntOrFloat] = None, step: Optional[IntOrFloat] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
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

    def add_gui_rgb(self, value: Tuple[int, int, int], label: str, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add an RGB picker to the GUI.

Args:
    value: Value of the RGBA picker.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def add_gui_rgba(self, value: Tuple[int, int, int, int], label: str, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
        """Add an RGBA picker to the GUI.

Args:
    value: Value of the RGBA picker.
    label: Label to display on the text input.
    disabled: Whether the text input is disabled.
    visible: Whether the text input is visible.
    hint: Optional hint to display on hover."""
        ...

    def add_gui_slider(self, value: IntOrFloat, label: str, min: IntOrFloat, max: IntOrFloat, step: IntOrFloat, marks: Optional[Tuple[IntOrFloat | Tuple[IntOrFloat, str], ...]] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
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

    def add_gui_tab_group(self, order: typing.Optional[float] = None, visible: bool = True):
        """Creates a new TabGroup instance

Args:
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def add_gui_tab_group_tab(self, label: str, order: typing.Optional[float] = None, visible: bool = True):
        """Handle for a tab in our visualizer.

    Lets us add GUI elements to it and remove it.

Args:
    label: Label to display on the tab.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def add_gui_text(self, value: str = '', order: typing.Optional[float] = None, visible: bool = True):
        """Creates a new Text instance

Args:
    value: Initial value of the text input.
    order: Optional ordering, smallest values will be displayed first.
    visible: Whether the component is visible."""
        ...

    def add_gui_vector2(self, value: Tuple[float, float] | np.ndarray, label: str, min: Tuple[float, float] | np.ndarray | None = None, max: Tuple[float, float] | np.ndarray | None = None, step: Optional[float] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
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

    def add_gui_vector3(self, value: Tuple[float, float, float] | np.ndarray, label: str, min: Tuple[float, float, float] | np.ndarray | None = None, max: Tuple[float, float, float] | np.ndarray | None = None, step: Optional[float] = None, disabled: bool = False, visible: bool = True, hint: Optional[str] = None):
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


def _apply_default_order(order: typing.Optional[float]) -> float:
    """Apply default ordering logic for GUI elements.

If `order` is set to a float, this function is a no-op and returns it back.
Otherwise, we increment and return the value of a global counter."""
    ...

def _default_factory(value):
    ...

def _get_container_i) -> str:
    """Get container ID associated with the current thread."""
    ...

def _get_param_docstring(docstring, name):
    ...

def _make_unique_i) -> str:
    """Return a unique ID for referencing GUI elements."""
    ...

def _pop_container_i) -> str:
    """Set container ID associated with the current thread."""
    ...

def _push_container_id(container_id: str) -> None:
    """Set container ID associated with the current thread."""
    ...

def _read_field_docstring):
    ...
