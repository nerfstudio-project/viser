# mypy: disable-error-code="misc"
#
# TLiteralString overloads are waiting on PEP 675 support in mypy.
# https://github.com/python/mypy/issues/12554
#
# In the meantime, it works great in Pyright/Pylance!

from __future__ import annotations

import abc
import base64
import contextlib
import io
import threading
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
import trimesh
import trimesh.visual
from typing_extensions import Literal, LiteralString, ParamSpec, TypeAlias, assert_never

from . import _messages, infra, theme
from ._gui import (
    GuiButtonGroupHandle,
    GuiButtonHandle,
    GuiDropdownHandle,
    GuiHandle,
    _GuiHandleState,
)
from ._scene_handle import (
    CameraFrustumHandle,
    FrameHandle,
    ImageHandle,
    LabelHandle,
    MeshHandle,
    PointCloudHandle,
    SceneNodeHandle,
    TransformControlsHandle,
    _SupportsVisibility,
    _TransformControlsState,
)

if TYPE_CHECKING:
    from .infra import ClientId


P = ParamSpec("P")


def _colors_to_uint8(colors: onp.ndarray) -> onpt.NDArray[onp.uint8]:
    """Convert intensity values to uint8. We assume the range [0,1] for floats, and
    [0,255] for integers."""
    if colors.dtype != onp.uint8:
        if onp.issubdtype(colors.dtype, onp.floating):
            colors = onp.clip(colors * 255.0, 0, 255).astype(onp.uint8)
        if onp.issubdtype(colors.dtype, onp.integer):
            colors = onp.clip(colors, 0, 255).astype(onp.uint8)
    return colors


RgbTupleOrArray: TypeAlias = Union[
    Tuple[int, int, int], Tuple[float, float, float], onp.ndarray
]


def _encode_rgb(rgb: RgbTupleOrArray) -> int:
    if isinstance(rgb, onp.ndarray):
        assert rgb.shape == (3,)
    rgb_fixed = tuple(
        value if onp.issubdtype(type(value), onp.integer) else int(value * 255)
        for value in rgb
    )
    assert len(rgb_fixed) == 3
    return int(rgb_fixed[0] * (256**2) + rgb_fixed[1] * 256 + rgb_fixed[2])


def _encode_image_base64(
    image: onp.ndarray,
    format: Literal["png", "jpeg"],
    jpeg_quality: Optional[int] = None,
) -> Tuple[Literal["image/png", "image/jpeg"], str]:
    media_type: Literal["image/png", "image/jpeg"]
    image = _colors_to_uint8(image)
    with io.BytesIO() as data_buffer:
        if format == "png":
            media_type = "image/png"
            iio.imwrite(data_buffer, image, extension=".png")
        elif format == "jpeg":
            media_type = "image/jpeg"
            iio.imwrite(
                data_buffer,
                image[..., :3],  # Strip alpha.
                extension=".jpeg",
                jpeg_quality=75 if jpeg_quality is None else jpeg_quality,
            )
        else:
            assert_never(format)

        base64_data = base64.b64encode(data_buffer.getvalue()).decode("ascii")

    return media_type, base64_data


def _make_gui_id() -> str:
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
    if x is None:
        return 1

    # Some risky float stuff...
    out = 0.0001
    while out < 1.0 and onp.abs(x) % (out * 10) < 1e-6:
        out = out * 10
    return out


def _compute_precision_digits(x: float) -> int:
    """For number inputs: compute digits of precision from some number.

    Example inputs/outputs:
        100 => 0
        0.5 => 1
        10.2 => 1
        0.007 => 3
    """
    precision = 0
    while onp.abs(int(x) - x) > 1e-7 or precision >= 7:
        x = x * 10
        precision += 1
    return precision


T = TypeVar("T")
TVector = TypeVar("TVector", bound=tuple)


def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if not isinstance(vector, tuple):
        assert cast(onp.ndarray, vector).shape == (length,)
    return cast(TVector, tuple(map(float, vector)))


IntOrFloat = TypeVar("IntOrFloat", int, float)
TString = TypeVar("TString", bound=str)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)


class MessageApi(abc.ABC):
    """Interface for all commands we can use to send messages over a websocket connection.

    Should be implemented by both our global server object (for broadcasting) and by
    invidividual clients."""

    def __init__(self, handler: infra.MessageHandler) -> None:
        self._gui_handle_state_from_id: Dict[str, _GuiHandleState[Any]] = {}
        self._handle_from_transform_controls_name: Dict[
            str, TransformControlsHandle
        ] = {}
        self._handle_from_node_name: Dict[str, SceneNodeHandle] = {}

        handler.register_handler(_messages.GuiUpdateMessage, self._handle_gui_updates)
        handler.register_handler(
            _messages.TransformControlsUpdateMessage,
            self._handle_transform_controls_updates,
        )
        handler.register_handler(
            _messages.SceneNodeClickedMessage,
            self._handle_click_updates,
        )

        self._gui_folder_labels: List[str] = []

        self._atomic_lock = threading.Lock()
        self._locked_thread_id = -1

    def configure_theme(
        self,
        *,
        titlebar_content: Optional[theme.TitlebarConfig] = None,
        fixed_sidebar: bool = False,
    ) -> None:
        """Configure the viser front-end's visual appearance."""
        self._queue(
            _messages.ThemeConfigurationMessage(
                titlebar_content=titlebar_content,
                fixed_sidebar=fixed_sidebar,
            ),
        )

    @contextlib.contextmanager
    def gui_folder(self, label: str) -> Generator[None, None, None]:
        """Context for placing all GUI elements into a particular folder. Folders can
        also be nested."""
        self._gui_folder_labels.append(label)
        yield
        assert self._gui_folder_labels.pop() == label

    def add_gui_button(
        self,
        label: str,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiButtonHandle:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`."""

        # Re-wrap the GUI handle with a button interface.
        id = _make_gui_id()
        return GuiButtonHandle(
            self._create_gui_input(
                initial_value=False,
                message=_messages.GuiAddButtonMessage(
                    order=time.time(),
                    id=id,
                    label=label,
                    folder_labels=tuple(self._gui_folder_labels),
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
        options: Iterable[TLiteralString],
        visible: bool = True,
        disabled: bool = False,
    ) -> GuiButtonGroupHandle[TLiteralString]:
        ...

    @overload
    def add_gui_button_group(
        self,
        label: str,
        options: Iterable[TString],
        visible: bool = True,
        disabled: bool = False,
    ) -> GuiButtonGroupHandle[TString]:
        ...

    def add_gui_button_group(
        self,
        label: str,
        options: Iterable[TLiteralString] | Iterable[TString],
        visible: bool = True,
        disabled: bool = False,
    ) -> GuiButtonGroupHandle[Any]:  # Return types are specified in overloads.
        """Add a button group to the GUI."""
        initial_value = next(iter(options))
        id = _make_gui_id()
        return GuiButtonGroupHandle(
            self._create_gui_input(
                initial_value,
                message=_messages.GuiAddButtonGroupMessage(
                    order=time.time(),
                    id=id,
                    label=label,
                    folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddCheckboxMessage(
                order=time.time(),
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddTextMessage(
                order=time.time(),
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
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

        id = _make_gui_id()
        return self._create_gui_input(
            initial_value=initial_value,
            message=_messages.GuiAddNumberMessage(
                order=time.time(),
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()

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
                folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()

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
                folder_labels=tuple(self._gui_folder_labels),
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
        options: Iterable[TLiteralString],
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
        options: Iterable[TString],
        initial_value: Optional[TString] = None,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiDropdownHandle[TString]:
        ...

    def add_gui_dropdown(
        self,
        label: str,
        options: Iterable[TLiteralString] | Iterable[TString],
        initial_value: Optional[TLiteralString | TString] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiDropdownHandle[Any]:  # Output type is specified in overloads.
        """Add a dropdown to the GUI."""
        if initial_value is None:
            initial_value = next(iter(options))
        id = _make_gui_id()
        return GuiDropdownHandle(
            self._create_gui_input(
                initial_value,
                message=_messages.GuiAddDropdownMessage(
                    order=time.time(),
                    id=id,
                    label=label,
                    folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()
        return self._create_gui_input(
            initial_value=initial_value,
            message=_messages.GuiAddSliderMessage(
                order=time.time(),
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddRgbMessage(
                order=time.time(),
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
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
        id = _make_gui_id()
        return self._create_gui_input(
            initial_value,
            message=_messages.GuiAddRgbaMessage(
                order=time.time(),
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
                initial_value=initial_value,
            ),
            disabled=disabled,
            visible=visible,
        )

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        color: RgbTupleOrArray = (20, 20, 20),
        image: Optional[onp.ndarray] = None,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: Optional[int] = None,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> CameraFrustumHandle:
        """Add a frustum to the scene. Useful for visualizing cameras.

        Like all cameras in the viser Python API, frustums follow the OpenCV [+Z forward,
        +X right, +Y down] convention.

        fov is vertical in radians; aspect is width over height."""

        if image is not None:
            media_type, base64_data = _encode_image_base64(
                image, format, jpeg_quality=jpeg_quality
            )
        else:
            media_type = None
            base64_data = None

        self._queue(
            _messages.CameraFrustumMessage(
                name=name,
                fov=fov,
                aspect=aspect,
                scale=scale,
                # (255, 255, 255) => 0xffffff, etc
                color=_encode_rgb(color),
                image_media_type=media_type,
                image_base64_data=base64_data,
            )
        )
        return CameraFrustumHandle._make(self, name, wxyz, position, visible)

    def add_frame(
        self,
        name: str,
        show_axes: bool = True,
        axes_length: float = 0.5,
        axes_radius: float = 0.025,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> FrameHandle:
        cast_vector(wxyz, length=4)
        cast_vector(position, length=3)
        self._queue(
            # TODO: remove wxyz and position from this message for consistency.
            _messages.FrameMessage(
                name=name,
                show_axes=show_axes,
                axes_length=axes_length,
                axes_radius=axes_radius,
            )
        )
        return FrameHandle._make(self, name, wxyz, position, visible)

    def add_label(
        self,
        name: str,
        text: str,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
    ) -> LabelHandle:
        """Add a 2D label to the scene."""
        self._queue(_messages.LabelMessage(name, text))
        return LabelHandle._make(self, name, wxyz, position)

    def add_point_cloud(
        self,
        name: str,
        points: onp.ndarray,
        colors: onp.ndarray,
        point_size: float = 0.1,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> PointCloudHandle:
        """Add a point cloud to the scene."""
        self._queue(
            _messages.PointCloudMessage(
                name=name,
                points=points.astype(onp.float32),
                colors=_colors_to_uint8(colors),
                point_size=point_size,
            )
        )
        return PointCloudHandle._make(self, name, wxyz, position, visible)

    def add_mesh(self, *args, **kwargs) -> MeshHandle:
        """Deprecated alias for `add_mesh_simple()`."""
        return self.add_mesh_simple(*args, **kwargs)

    def add_mesh_simple(
        self,
        name: str,
        vertices: onp.ndarray,
        faces: onp.ndarray,
        color: RgbTupleOrArray = (90, 200, 255),
        wireframe: bool = False,
        side: Literal["front", "back", "double"] = "front",
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshHandle:
        """Add a mesh to the scene."""
        self._queue(
            _messages.MeshMessage(
                name,
                vertices.astype(onp.float32),
                faces.astype(onp.uint32),
                # (255, 255, 255) => 0xffffff, etc
                color=_encode_rgb(color),
                vertex_colors=None,
                wireframe=wireframe,
                side=side,
            )
        )
        node_handle = MeshHandle._make(self, name, wxyz, position, visible)
        return node_handle

    def add_mesh_trimesh(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        wireframe: bool = False,
        side: Literal["front", "back", "double"] = "front",
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshHandle:
        """Add a trimesh mesh to the scene."""
        if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
            vertex_colors = mesh.visual.vertex_colors
            self._queue(
                _messages.MeshMessage(
                    name,
                    mesh.vertices.astype(onp.float32),
                    mesh.faces.astype(onp.uint32),
                    color=None,
                    vertex_colors=(
                        vertex_colors.view(onp.ndarray).astype(onp.uint8)[..., :3]
                    ),
                    wireframe=wireframe,
                    side=side,
                )
            )
        elif isinstance(mesh.visual, trimesh.visual.TextureVisuals):
            # TODO: this needs to be implemented.
            import warnings

            warnings.warn(
                "Texture visuals are not fully supported yet!",
                stacklevel=2,
            )
            self._queue(
                _messages.MeshMessage(
                    name,
                    mesh.vertices.astype(onp.float32),
                    mesh.faces.astype(onp.uint32),
                    color=_encode_rgb(
                        # Note that `vertex_colors` here is per-UV coordinate, not
                        # per mesh vertex.
                        mesh.visual.to_color().vertex_colors.flatten()[:3]
                    ),
                    vertex_colors=(None),
                    wireframe=wireframe,
                    side=side,
                )
            )
        else:
            assert False, f"Unsupported texture visuals: {mesh.visual}"

        return MeshHandle._make(self, name, wxyz, position, visible)

    def set_background_image(
        self,
        image: onp.ndarray,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: Optional[int] = None,
    ) -> None:
        """Set a background image for the scene. Useful for NeRF visualization."""
        media_type, base64_data = _encode_image_base64(
            image, format, jpeg_quality=jpeg_quality
        )
        self._queue(
            _messages.BackgroundImageMessage(
                media_type=media_type, base64_data=base64_data
            )
        )

    def add_image(
        self,
        name: str,
        image: onp.ndarray,
        render_width: float,
        render_height: float,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: Optional[int] = None,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> ImageHandle:
        """Add a 2D image to the scene. Rendered in 3D."""
        media_type, base64_data = _encode_image_base64(
            image, format, jpeg_quality=jpeg_quality
        )
        self._queue(
            _messages.ImageMessage(
                name=name,
                media_type=media_type,
                base64_data=base64_data,
                render_width=render_width,
                render_height=render_height,
            )
        )
        return ImageHandle._make(self, name, wxyz, position, visible)

    def add_transform_controls(
        self,
        name: str,
        scale: float = 1.0,
        line_width: float = 2.5,
        fixed: bool = False,
        auto_transform: bool = True,
        active_axes: Tuple[bool, bool, bool] = (True, True, True),
        disable_axes: bool = False,
        disable_sliders: bool = False,
        disable_rotations: bool = False,
        translation_limits: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ] = ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        rotation_limits: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ] = ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        depth_test: bool = True,
        opacity: float = 1.0,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> TransformControlsHandle:
        """Add a transform gizmo for interacting with the scene."""
        # That decorator factory would be really helpful here...
        self._queue(
            _messages.TransformControlsMessage(
                name=name,
                scale=scale,
                line_width=line_width,
                fixed=fixed,
                auto_transform=auto_transform,
                active_axes=active_axes,
                disable_axes=disable_axes,
                disable_sliders=disable_sliders,
                disable_rotations=disable_rotations,
                translation_limits=translation_limits,
                rotation_limits=rotation_limits,
                depth_test=depth_test,
                opacity=opacity,
            )
        )

        def sync_cb(client_id: ClientId, state: TransformControlsHandle) -> None:
            message_orientation = _messages.SetOrientationMessage(
                name=name,
                wxyz=tuple(map(float, state._impl.wxyz)),  # type: ignore
            )
            message_orientation.excluded_self_client = client_id
            self._queue(message_orientation)

            message_position = _messages.SetPositionMessage(
                name=name,
                position=tuple(map(float, state._impl.position)),  # type: ignore
            )
            message_position.excluded_self_client = client_id
            self._queue(message_position)

        node_handle = _SupportsVisibility._make(self, name, wxyz, position, visible)
        state_aux = _TransformControlsState(
            last_updated=time.time(),
            update_cb=[],
            sync_cb=sync_cb,
        )
        handle = TransformControlsHandle(node_handle._impl, state_aux)
        self._handle_from_transform_controls_name[name] = handle
        return handle

    def reset_scene(self):
        """Reset the scene."""
        self._queue(_messages.ResetSceneMessage())

    @abc.abstractmethod
    def _queue(self, message: _messages.Message) -> None:
        """Abstract method for sending messages."""
        ...

    def _handle_gui_updates(
        self, client_id: ClientId, message: _messages.GuiUpdateMessage
    ) -> None:
        """Callback for handling GUI messages."""
        handle_state = self._gui_handle_state_from_id.get(message.id, None)
        if handle_state is None:
            return

        value = handle_state.typ(message.value)

        # Only call update when value has actually changed.
        if not handle_state.is_button and value == handle_state.value:
            return

        # Update state.
        with self._atomic_lock:
            handle_state.value = value
            handle_state.update_timestamp = time.time()

        # Trigger callbacks.
        for cb in handle_state.update_cb:
            cb(GuiHandle(handle_state))
        if handle_state.sync_cb is not None:
            handle_state.sync_cb(client_id, value)

    def _handle_transform_controls_updates(
        self, client_id: ClientId, message: _messages.TransformControlsUpdateMessage
    ) -> None:
        """Callback for handling transform gizmo messages."""
        handle = self._handle_from_transform_controls_name.get(message.name, None)
        if handle is None:
            return

        # Update state.
        handle._impl.wxyz = onp.array(message.wxyz)
        handle._impl.position = onp.array(message.position)
        handle._impl_aux.last_updated = time.time()

        # Trigger callbacks.
        for cb in handle._impl_aux.update_cb:
            cb(handle)
        if handle._impl_aux.sync_cb is not None:
            handle._impl_aux.sync_cb(client_id, handle)

    def _handle_click_updates(
        self, client_id: ClientId, message: _messages.SceneNodeClickedMessage
    ) -> None:
        """Callback for handling click messages."""
        handle = self._handle_from_node_name.get(message.name, None)
        if handle is None or handle._impl.click_cb is None:
            return
        for cb in handle._impl.click_cb:
            cb(handle)

    def _create_gui_input(
        self,
        initial_value: T,
        message: _messages._GuiAddMessageBase,
        disabled: bool,
        visible: bool,
        is_button: bool = False,
    ) -> GuiHandle[T]:
        """Private helper for adding a simple GUI element."""

        # Send add GUI input message.
        self._queue(message)

        # Construct handle.
        handle_state = _GuiHandleState(
            label=message.label,
            typ=type(initial_value),
            api=self,
            value=initial_value,
            update_timestamp=time.time(),
            folder_labels=message.folder_labels,
            update_cb=[],
            is_button=is_button,
            sync_cb=None,
            cleanup_cb=None,
            disabled=False,
            visible=True,
            id=message.id,
            order=message.order,
            initial_value=initial_value,
        )
        self._gui_handle_state_from_id[handle_state.id] = handle_state
        handle_state.cleanup_cb = lambda: self._gui_handle_state_from_id.pop(
            handle_state.id
        )

        # For broadcasted GUI handles, we should synchronize all clients.
        # This will be a no-op for client handles.
        if not is_button:

            def sync_other_clients(client_id: ClientId, value: Any) -> None:
                message = _messages.GuiSetValueMessage(id=handle_state.id, value=value)
                message.excluded_self_client = client_id
                self._queue(message)

            handle_state.sync_cb = sync_other_clients

        handle = GuiHandle(handle_state)

        # Set the disabled/visible fields. These will queue messages under-the-hood.
        if disabled:
            handle.disabled = disabled
        if not visible:
            handle.visible = visible

        return handle
