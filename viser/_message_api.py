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
    Callable,
    Dict,
    Generator,
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
        id = str(uuid.uuid4())
        self._queue(
            _messages.GuiAddButtonMessage(
                order=0,
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
            )
        )
        return GuiButtonHandle(
            self._add_gui_impl(
                id,
                initial_value=False,
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
        name: str,
        options: List[TLiteralString],
        visible: bool = True,
    ) -> GuiButtonGroupHandle[TLiteralString]:
        ...

    @overload
    def add_gui_button_group(
        self,
        name: str,
        options: List[TString],
        visible: bool = True,
    ) -> GuiButtonGroupHandle[TString]:
        ...

    def add_gui_button_group(
        self,
        name: str,
        options: List[TLiteralString] | List[TString],
        visible: bool = True,
    ) -> GuiButtonGroupHandle[Any]:  # Return types are specified in overloads.
        """Add a button group to the GUI. Button groups currently cannot be disabled."""
        assert len(options) > 0
        assert False
        # handle = self._add_gui_impl(
        #     name,
        #     options[0],
        #     leva_conf={
        #         "type": "BUTTON_GROUP",
        #         "label": name,
        #         "opts": options,
        #     },
        #     disabled=False,
        #     visible=visible,
        #     is_button=True,
        # )
        #
        # # Re-wrap the GUI handle with a button group interface.
        # return GuiButtonGroupHandle(_impl=handle._impl)

    def add_gui_checkbox(
        self,
        name: str,
        initial_value: bool,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[bool]:
        """Add a checkbox to the GUI."""
        assert isinstance(initial_value, bool)
        pass
        # assert False
        # return self._add_gui_impl(
        #     "/".join(self._gui_folder_labels + [name]),
        #     initial_value,
        #     leva_conf={"value": initial_value, "label": name},
        #     disabled=disabled,
        #     visible=visible,
        #     hint=hint,
        # )

    def add_gui_text(
        self,
        name: str,
        initial_value: str,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[str]:
        """Add a text input to the GUI."""
        assert isinstance(initial_value, str)
        # assert False
        pass
        # return self._add_gui_impl(
        #     "/".join(self._gui_folder_labels + [name]),
        #     initial_value,
        #     leva_conf={"value": initial_value, "label": name},
        #     disabled=disabled,
        #     visible=visible,
        #     hint=hint,
        # )

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

            def _compute_step(x: Optional[float]) -> float:  # type: ignore
                if x is None:
                    return 1

                # Some risky float stuff...
                out = 0.0001
                while out < 1.0 and onp.abs(x) % (out * 10) < 1e-6:
                    out = out * 10
                return out

            step = float(  # type: ignore
                onp.min(
                    [
                        _compute_step(initial_value),
                        _compute_step(min),
                        _compute_step(max),
                    ]
                )
            )

        # Determine precision from step.
        assert step is not None
        x = step
        precision = 0
        while onp.abs(int(x) - x) > 1e-7:
            x = x * 10
            precision += 1

        # Re-wrap the GUI handle with a button interface.
        id = str(uuid.uuid4())
        self._queue(
            _messages.GuiAddNumberMessage(
                order=0,
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
                initial_value=initial_value,
                min=min,
                max=max,
                precision=precision,
                step=step,
            )
        )
        return self._add_gui_impl(
            id,
            initial_value=initial_value,
            disabled=disabled,
            visible=visible,
            is_button=False,
        )

    def add_gui_vector2(
        self,
        name: str,
        initial_value: Tuple[float, float] | onp.ndarray,
        step: Optional[float] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI."""
        pass
        # return self._add_gui_impl(
        #     "/".join(self._gui_folder_labels + [name]),
        #     cast_vector(initial_value, length=2),
        #     leva_conf={
        #         "value": initial_value,
        #         "label": name,
        #         "step": step,
        #     },
        #     disabled=disabled,
        #     visible=visible,
        #     hint=hint,
        # )

    def add_gui_vector3(
        self,
        name: str,
        initial_value: Tuple[float, float, float] | onp.ndarray,
        step: Optional[float] = None,
        lock: bool = False,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI."""
        pass
        # return self._add_gui_impl(
        #     "/".join(self._gui_folder_labels + [name]),
        #     cast_vector(initial_value, length=3),
        #     leva_conf={
        #         "label": name,
        #         "value": initial_value,
        #         "step": step,
        #         "lock": lock,
        #     },
        #     disabled=disabled,
        #     visible=visible,
        #     hint=hint,
        # )

    # See add_gui_dropdown for notes on overloads.
    @overload
    def add_gui_dropdown(
        self,
        name: str,
        options: List[TLiteralString],
        initial_value: Optional[TLiteralString] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiDropdownHandle[TLiteralString]:
        ...

    @overload
    def add_gui_dropdown(
        self,
        name: str,
        options: List[TString],
        initial_value: Optional[TString] = None,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiDropdownHandle[TString]:
        ...

    def add_gui_dropdown(
        self,
        name: str,
        options: List[TLiteralString] | List[TString],
        initial_value: Optional[TLiteralString | TString] = None,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiDropdownHandle[Any]:  # Output type is specified in overloads.
        """Add a dropdown to the GUI."""
        assert len(options) > 0
        assert False
        # if initial_value is None:
        #     initial_value = options[0]
        #
        # # Re-wrap the GUI handle with a select interface.
        # return GuiDropdownHandle(
        #     self._add_gui_impl(
        #         "/".join(self._gui_folder_labels + [name]),
        #         initial_value,
        #         leva_conf={
        #             "value": initial_value,
        #             "label": name,
        #             "options": options,
        #         },
        #         disabled=disabled,
        #         visible=visible,
        #         hint=hint,
        #     )._impl,
        #     options,
        # )

    def add_gui_slider(
        self,
        label: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: Optional[IntOrFloat],
        initial_value: IntOrFloat,
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[IntOrFloat]:
        """Add a slider to the GUI. Types of the min, max, step, and initial value should match."""
        assert max >= min
        if step is not None and step > max - min:
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
        id = str(uuid.uuid4())
        self._queue(
            _messages.GuiAddSliderMessage(
                order=0,
                id=id,
                label=label,
                folder_labels=tuple(self._gui_folder_labels),
                min=min,
                max=max,
                step=step,
                initial_value=initial_value,
            )
        )
        return self._add_gui_impl(
            id,
            initial_value=initial_value,
            disabled=disabled,
            visible=visible,
            is_button=False,
        )

    def add_gui_rgb(
        self,
        name: str,
        initial_value: Tuple[int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[int, int, int]]:
        """Add an RGB picker to the GUI."""
        assert False
        # return self._add_gui_impl(
        #     "/".join(self._gui_folder_labels + [name]),
        #     initial_value,
        #     leva_conf={
        #         "value": {
        #             "r": initial_value[0],
        #             "g": initial_value[1],
        #             "b": initial_value[2],
        #         },
        #         "label": name,
        #     },
        #     disabled=disabled,
        #     visible=visible,
        #     encoder=lambda rgb: dict(zip("rgb", rgb)),
        #     decoder=lambda rgb_dict: (rgb_dict["r"], rgb_dict["g"], rgb_dict["b"]),
        #     hint=hint,
        # )

    def add_gui_rgba(
        self,
        name: str,
        initial_value: Tuple[int, int, int, int],
        disabled: bool = False,
        visible: bool = True,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[int, int, int, int]]:
        """Add an RGBA picker to the GUI."""
        assert False
        # return self._add_gui_impl(
        #     "/".join(self._gui_folder_labels + [name]),
        #     initial_value,
        #     leva_conf={
        #         "value": {
        #             "r": initial_value[0],
        #             "g": initial_value[1],
        #             "b": initial_value[2],
        #             "a": initial_value[3],
        #         },
        #         "label": name,
        #     },
        #     disabled=disabled,
        #     visible=visible,
        #     encoder=lambda rgba: dict(zip("rgba", rgba)),
        #     decoder=lambda rgba_dict: (
        #         rgba_dict["r"],
        #         rgba_dict["g"],
        #         rgba_dict["b"],
        #         rgba_dict["a"],
        #     ),
        #     hint=hint,
        # )

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

        value = handle_state.typ(handle_state.decoder(message.value))

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

    def _add_gui_impl(
        self,
        id: str,
        initial_value: T,
        disabled: bool,
        visible: bool,
        is_button: bool = False,
        encoder: Callable[[T], Any] = lambda x: x,
        decoder: Callable[[Any], T] = lambda x: x,
        hint: Optional[str] = None,
    ) -> GuiHandle[T]:
        """Private helper for adding a simple GUI element."""

        handle_state = _GuiHandleState(
            id,
            typ=type(initial_value),
            api=self,
            value=initial_value,
            update_timestamp=time.time(),
            folder_labels=self._gui_folder_labels,
            update_cb=[],
            is_button=is_button,
            sync_cb=None,
            cleanup_cb=None,
            encoder=encoder,
            decoder=decoder,
            disabled=False,
            visible=True,
        )
        self._gui_handle_state_from_id[id] = handle_state
        handle_state.cleanup_cb = lambda: self._gui_handle_state_from_id.pop(id)

        # For broadcasted GUI handles, we should synchronize all clients.
        # This will be a no-op for client handles.
        if not is_button:

            def sync_other_clients(client_id: ClientId, value: Any) -> None:
                message = _messages.GuiSetValueMessage(
                    id=id, value=handle_state.encoder(value)
                )
                message.excluded_self_client = client_id
                self._queue(message)

            handle_state.sync_cb = sync_other_clients

            # Explicitly set the initial value of the GUI element. This is needed when
            # the GUI element already exists on the front-end's Leva store.
            self._queue(_messages.GuiSetValueMessage(id=id, value=initial_value))

        handle = GuiHandle(handle_state)

        # Set the disabled/visible fields. These will queue messages under-the-hood.
        if disabled:
            handle.disabled = disabled
        if not visible:
            handle.visible = visible

        return handle
