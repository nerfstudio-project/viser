from dataclasses import field, InitVar
from functools import wraps
import time
from typing import Optional, Literal, Union, TypeVar, Generic, Tuple, Type
from typing import Callable, Any
from dataclasses import dataclass
try:
    from typing import Concatenate
except ImportError:
    from typing_extensions import Concatenate
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec


TProps = TypeVar("TProps")
TReturn = TypeVar('TReturn')
TArgs = ParamSpec('TArgs')
T = TypeVar("T")


def copy_signature(fn_signature: Callable[TArgs, Any]):
    def wrapper(fn: Callable[..., TReturn]) -> Callable[Concatenate[Any, TArgs], TReturn]:
        out = wraps(fn_signature)(fn)
        # TODO: perhaps copy signature from fn_signature and get help for arguments
        out.__doc__ = f"""Creates a new GUI {fn_signature.__name__} component and returns a handle to it.

Returns:
    The component handle.
"""
        return out
    return wrapper


class Property(Generic[T]):
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def get(self) -> T:
        return self.getter()

    def set(self, value: T):
        self.setter(value)


@dataclass(kw_only=True)
class GuiComponent(Protocol):
    order: InitVar[Optional[float]] = None

    @property
    def order(self) -> Optional[float]:
        return object.__getattribute__(self, "_order")

    @property
    def id(self):
        raise NotImplementedError()

    def __post_init__(self, order: Optional[float]):
        object.__setattr__(self, "_order", order)

    def property(self, name: str) -> Property[T]:
        raise NotImplementedError()


@dataclass(kw_only=True)
class Button(GuiComponent, Protocol):
    """Button component
    """
    label: str
    """Button label"""
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
    ] = None
    """Button color"""
    icon_base64: Optional[str] = None
    """Icon to display on the button, as a base64-encoded SVG image."""
    disabled: bool = False
    """Whether the button is disabled."""
    hint: Optional[str] = None
    """Button tooltip."""


@dataclass(kw_only=True)
class Input(GuiComponent, Protocol):
    value: str
    label: str
    hint: Optional[str]
    disabled: bool = False


@dataclass(kw_only=True)
class TextInput(Input, Protocol):
    pass

@dataclass(kw_only=True)
class Folder(GuiComponent, Protocol):
    label: str
    expand_by_default: bool = True

@dataclass(kw_only=True)
class Markdown(GuiComponent, Protocol):
    markdown: str

@dataclass(kw_only=True)
class TabGroup(GuiComponent, Protocol):
    tab_labels: Tuple[str, ...]
    tab_icons_base64: Tuple[Union[str, None], ...]
    tab_container_ids: Tuple[str, ...]

@dataclass(kw_only=True)
class Modal(GuiComponent, Protocol):
    order: float
    id: str
    title: str


@dataclass(kw_only=True)
class Slider(Input, Protocol):
    value: float
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    precision: Optional[int] = None


@dataclass(kw_only=True)
class NumberInput(Input, Protocol):
    value: float
    step: float
    min: Optional[float] = None
    max: Optional[float] = None
    precision: Optional[int] = None


@dataclass(kw_only=True)
class RgbInput(Input, Protocol):
    value: Tuple[int, int, int]


@dataclass(kw_only=True)
class RgbaInput(Input, Protocol):
    value: Tuple[int, int, int, int]


@dataclass(kw_only=True)
class Checkbox(Input, Protocol):
    value: bool


@dataclass(kw_only=True)
class Vector2Input(Input, Protocol):
    value: Tuple[float, float]
    step: float
    min: Optional[Tuple[float, float]] = None
    max: Optional[Tuple[float, float]] = None
    precision: Optional[int] = None


@dataclass(kw_only=True)
class Vector3Input(Input, Protocol):
    value: Tuple[float, float, float]
    min: Optional[Tuple[float, float, float]]
    max: Optional[Tuple[float, float, float]]
    step: float
    precision: int


@dataclass(kw_only=True)
class Dropdown(Input, Protocol):
    options: Tuple[str, ...]
    value: Optional[str] = None


class GuiApiMixin:
    @copy_signature(Button)
    def add_gui_button(self, *args, **kwargs) -> Button:
        props = Button(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(TextInput)
    def gui_add_text_input(self, *args, **kwargs) -> TextInput:
        props = TextInput(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(NumberInput)
    def add_gui_number(self, *args, **kwargs) -> NumberInput:
        props = NumberInput(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Modal)
    def add_gui_modal(self, *args, **kwargs) -> Modal:
        props = Modal(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Slider)
    def add_gui_slider(self, *args, **kwargs) -> Slider:
        props = Slider(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Checkbox)
    def add_gui_checkbox(self, *args, **kwargs) -> Checkbox:
        props = Checkbox(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(RgbInput)
    def add_gui_rgb(self, *args, **kwargs) -> RgbInput:
        props = RgbInput(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(RgbaInput)
    def add_gui_rgba(self, *args, **kwargs) -> RgbaInput:
        props = RgbaInput(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Folder)
    def add_gui_folder(self, *args, **kwargs) -> Folder:
        props = Folder(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Markdown)
    def add_gui_markdown(self, *args, **kwargs) -> Markdown:
        props = Markdown(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(TabGroup)
    def add_gui_tab_group(self, *args, **kwargs) -> TabGroup:
        props = TabGroup(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Vector2Input)
    def add_gui_vector2(self, *args, **kwargs) -> Vector2Input:
        props = Vector2Input(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Vector3Input)
    def add_gui_vector3(self, *args, **kwargs) -> Vector3Input:
        props = Vector3Input(*args, **kwargs)
        return self.gui_add_component(props)

    @copy_signature(Dropdown)
    def add_gui_dropdown(self, *args, **kwargs) -> Dropdown:
        props = Dropdown(*args, **kwargs)
        return self.gui_add_component(props)

    def gui_add_component(self, props: TProps) -> TProps:
        raise NotImplementedError()


Component = Union[
    Button,
    TextInput,
    NumberInput,
    Slider,
    Checkbox,
    RgbInput,
    RgbaInput,
    Folder,
    Markdown,
    TabGroup,
    Modal,
    Vector2Input,
    Vector3Input,
    Dropdown,
]
