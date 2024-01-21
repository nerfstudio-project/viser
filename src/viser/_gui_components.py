from dataclasses import field
from functools import wraps
import time
from typing import Optional, Literal, Union, TypeVar, Generic, Tuple
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
        return wraps(fn_signature)(fn)
    return wrapper


class Property(Generic[T]):
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def get(self) -> T:
        return self.getter()

    def set(self, value: T):
        self.setter(value)


class ComponentHandle(Generic[TProps]):
    def __init__(self, update, id: str, props: TProps):
        self.id = id
        self._props = props
        self._api_update = update
        self._update_timestamp = time.time()

    def _update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self._props, k):
                raise AttributeError(f"Component has no property {k}")
            setattr(self._props, k, v)
        self._update_timestamp = time.time()

        # Raise message to update component.
        self._api_update(self._impl.id, kwargs)

    def property(self, name: str) -> Property[T]:
        if not hasattr(self._props, name):
            raise AttributeError(f"Component has no property {name}")
        return Property(
            lambda: getattr(self._props, name),
            lambda value: self._update(**{name: value}),
        )

    def __getattribute__(self, name: str) -> T:
        if not hasattr(ComponentHandle, name):
            return self.property(name).get()
        else:
            return super().__getattribute__(name)


@dataclass
class Button(Protocol):
    label: str
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
    icon_base64: Optional[str] = None
    disabled: bool = False
    hint: Optional[str] = None


@dataclass
class Input(Protocol):
    value: str
    label: str
    hint: Optional[str]
    disabled: bool = False


@dataclass
class TextInput(Input):
    pass

@dataclass
class Folder(Protocol):
    label: str
    expand_by_default: bool = True

@dataclass
class Markdown(Protocol):
    markdown: str

@dataclass
class TabGroup(Protocol):
    tab_labels: Tuple[str, ...]
    tab_icons_base64: Tuple[Union[str, None], ...]
    tab_container_ids: Tuple[str, ...]

@dataclass
class Modal(Protocol):
    order: float
    id: str
    title: str


@dataclass(kw_only=True)
class Slider(Input):
    value: float
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    precision: Optional[int] = None


@dataclass(kw_only=True)
class NumberInput(Input):
    value: float
    step: float
    min: Optional[float] = None
    max: Optional[float] = None
    precision: Optional[int] = None


@dataclass(kw_only=True)
class RgbInput(Input):
    value: Tuple[int, int, int]


@dataclass(kw_only=True)
class RgbaInput(Input):
    value: Tuple[int, int, int, int]


@dataclass(kw_only=True)
class Checkbox(Input):
    value: bool


@dataclass(kw_only=True)
class Vector2Input(Input):
    value: Tuple[float, float]
    step: float
    min: Optional[Tuple[float, float]] = None
    max: Optional[Tuple[float, float]] = None
    precision: Optional[int] = None


@dataclass(kw_only=True)
class Vector3Input(Input):
    value: Tuple[float, float, float]
    min: Optional[Tuple[float, float, float]]
    max: Optional[Tuple[float, float, float]]
    step: float
    precision: int


@dataclass(kw_only=True)
class Dropdown(Input):
    options: Tuple[str, ...]
    value: Optional[str] = None


# Create handles for each component.
# This is a workaround for Python's lack of support for
# type intersection. See:
# https://github.com/python/typing/issues/18

class ButtonHandle(ComponentHandle[Button], Button):
    pass


class TextInputHandle(ComponentHandle[TextInput], TextInput):
    pass


class NumberInputHandle(ComponentHandle[NumberInput], NumberInput):
    pass


class SliderHandle(ComponentHandle[Slider], Slider):
    pass


class CheckboxHandle(ComponentHandle[Checkbox], Checkbox):
    pass


class RgbInputHandle(ComponentHandle[RgbInput], RgbInput):
    pass


class RgbaInputHandle(ComponentHandle[RgbaInput], RgbaInput):
    pass


class FolderHandle(ComponentHandle[Folder], Folder):
    pass


class MarkdownHandle(ComponentHandle[Markdown], Markdown):
    pass


class TabGroupHandle(ComponentHandle[TabGroup], TabGroup):
    pass


class ModalHandle(ComponentHandle[Modal], Modal):
    pass


class Vector2InputHandle(ComponentHandle[Vector2Input], Vector2Input):
    pass


class Vector3InputHandle(ComponentHandle[Vector3Input], Vector3Input):
    pass


class DropdownHandle(ComponentHandle[Dropdown], Dropdown):
    pass


# This could also be done with typing, but not at the moment:
# https://peps.python.org/pep-0612/#concatenating-keyword-parameters
# For now we have to copy the signature manually.
@dataclass
class _AddComponentProtocol(Protocol):
    order: Optional[float] = field(default=None, kw_only=True)

    @staticmethod
    def split_kwargs(kwargs):
        kwargs = kwargs.copy()
        main_kwargs = {}
        for k in kwargs:
            if k in vars(_AddComponentProtocol):
                main_kwargs[k] = kwargs.pop(k)
        return main_kwargs, kwargs

@dataclass
class _AddButtonProtocol(_AddComponentProtocol, Button):
    pass


@dataclass
class _AddTextInputProtocol(_AddComponentProtocol, TextInput):
    pass

@dataclass
class _AddNumberInputProtocol(_AddComponentProtocol, NumberInput):
    pass

@dataclass
class _AddSliderProtocol(_AddComponentProtocol, Slider):
    pass


@dataclass
class _AddCheckboxProtocol(_AddComponentProtocol, Checkbox):
    pass


@dataclass
class _AddRgbInputProtocol(_AddComponentProtocol, RgbInput):
    pass


@dataclass
class _AddRgbaInputProtocol(_AddComponentProtocol, RgbaInput):
    pass


@dataclass
class _AddFolderProtocol(_AddComponentProtocol, Folder):
    pass


@dataclass
class _AddMarkdownProtocol(_AddComponentProtocol, Markdown):
    pass


@dataclass
class _AddTabGroupProtocol(_AddComponentProtocol, TabGroup):
    pass


@dataclass
class _AddModalProtocol(_AddComponentProtocol, Modal):
    pass


@dataclass
class _AddVector2InputProtocol(_AddComponentProtocol, Vector2Input):
    pass


@dataclass
class _AddVector3InputProtocol(_AddComponentProtocol, Vector3Input):
    pass


@dataclass
class _AddDropdownProtocol(_AddComponentProtocol, Dropdown):
    pass


class GuiApiMixin:
    @copy_signature(_AddButtonProtocol)
    def add_gui_button(self, *args, **kwargs) -> ButtonHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Button(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddTextInputProtocol)
    def gui_add_text_input(self, *args, **kwargs) -> TextInputHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(TextInput(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddNumberInputProtocol)
    def add_gui_number(self, *args, **kwargs) -> NumberInputHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(NumberInput(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddModalProtocol)
    def add_gui_modal(self, *args, **kwargs) -> ModalHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Modal(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddSliderProtocol)
    def add_gui_slider(self, *args, **kwargs) -> SliderHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Slider(*args, **kwargs), order=main_kwargs)


    @copy_signature(_AddCheckboxProtocol)
    def add_gui_checkbox(self, *args, **kwargs) -> CheckboxHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Checkbox(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddRgbInputProtocol)
    def add_gui_rgb(self, *args, **kwargs) -> RgbInputHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(RgbInput(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddRgbaInputProtocol)
    def add_gui_rgba(self, *args, **kwargs) -> RgbaInputHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(RgbaInput(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddFolderProtocol)
    def add_gui_folder(self, *args, **kwargs) -> FolderHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Folder(*args, **kwargs), order=main_kwargs)

    @copy_signature(_AddMarkdownProtocol)
    def add_gui_markdown(self, *args, **kwargs) -> MarkdownHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Markdown(*args, **kwargs), order=main_kwargs)


    @copy_signature(_AddTabGroupProtocol)
    def add_gui_tab_group(self, *args, **kwargs) -> TabGroupHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(TabGroup(*args, **kwargs), order=main_kwargs)


    @copy_signature(_AddVector2InputProtocol)
    def add_gui_vector2(self, *args, **kwargs) -> Vector2InputHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Vector2Input(*args, **kwargs), order=main_kwargs)


    @copy_signature(_AddVector3InputProtocol)
    def add_gui_vector3(self, *args, **kwargs) -> Vector3InputHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Vector3Input(*args, **kwargs), order=main_kwargs)


    @copy_signature(_AddDropdownProtocol)
    def add_gui_dropdown(self, *args, **kwargs) -> DropdownHandle:
        main_kwargs, kwargs = _AddComponentProtocol.split_kwargs(kwargs)
        self.gui_add_component(Dropdown(*args, **kwargs), order=main_kwargs)


    def gui_add_component(self, component: Any, order: Optional[float] = None):
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