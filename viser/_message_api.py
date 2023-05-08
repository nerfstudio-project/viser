# mypy: disable-error-code="misc"
#
# TLiteralString overload on `add_gui_select()` is waiting on PEP 675 support in mypy.
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
    cast,
    overload,
)

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
from typing_extensions import Literal, LiteralString, ParamSpec, assert_never

from . import _messages, infra
from ._gui import GuiButtonHandle, GuiHandle, GuiSelectHandle, _GuiHandleState
from ._scene_handle import (
    SceneNodeHandle,
    TransformControlsHandle,
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


def _encode_rgb(
    rgb: Tuple[int, int, int]
    | Tuple[float, float, float]
    | onp.ndarray = (80, 120, 255),
) -> int:
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
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)


class MessageApi(abc.ABC):
    """Interface for all commands we can use to send messages over a websocket connection.

    Should be implemented by both our global server object (for broadcasting) and by
    invidividual clients."""

    def __init__(self, handler: infra.MessageHandler) -> None:
        self._handle_state_from_gui_name: Dict[str, _GuiHandleState[Any]] = {}
        self._handle_from_transform_controls_name: Dict[
            str, TransformControlsHandle
        ] = {}

        handler.register_handler(_messages.GuiUpdateMessage, self._handle_gui_updates)
        handler.register_handler(
            _messages.TransformControlsUpdateMessage,
            self._handle_transform_controls_updates,
        )

        self._gui_folder_labels: List[str] = []

        self._atomic_lock = threading.Lock()
        self._locked_thread_id = -1

    @contextlib.contextmanager
    def gui_folder(self, label: str) -> Generator[None, None, None]:
        """Context for placing all GUI elements into a particular folder. Folders can
        also be nested."""
        self._gui_folder_labels.append(label)
        yield
        assert self._gui_folder_labels.pop() == label

    def add_gui_button(
        self,
        name: str,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiButtonHandle:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`.

        Currently, all button names need to be unique."""
        return GuiButtonHandle(
            self._add_gui_impl(
                name,
                initial_value=False,
                leva_conf={"type": "BUTTON", "settings": {}},
                disabled=disabled,
                visible=visible,
                is_button=True,
            )._impl
        )

    def add_gui_checkbox(
        self,
        name: str,
        initial_value: bool,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[bool]:
        """Add a checkbox to the GUI."""
        assert isinstance(initial_value, bool)
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={"value": initial_value, "label": name},
            disabled=disabled,
            visible=visible,
        )

    def add_gui_text(
        self,
        name: str,
        initial_value: str,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[str]:
        """Add a text input to the GUI."""
        assert isinstance(initial_value, str)
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={"value": initial_value, "label": name},
            disabled=disabled,
            visible=visible,
        )

    def add_gui_number(
        self,
        name: str,
        initial_value: IntOrFloat,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[IntOrFloat]:
        """Add a number input to the GUI."""
        assert isinstance(initial_value, (int, float))
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={"value": initial_value, "label": name},
            disabled=disabled,
            visible=visible,
        )

    def add_gui_vector2(
        self,
        name: str,
        initial_value: Tuple[float, float] | onp.ndarray,
        step: Optional[float] = None,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI."""
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            cast_vector(initial_value, length=2),
            leva_conf={
                "value": initial_value,
                "label": name,
                "step": step,
            },
            disabled=disabled,
            visible=visible,
        )

    def add_gui_vector3(
        self,
        name: str,
        initial_value: Tuple[float, float, float] | onp.ndarray,
        step: Optional[float] = None,
        lock: bool = False,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI."""
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            cast_vector(initial_value, length=3),
            leva_conf={
                "label": name,
                "value": initial_value,
                "step": step,
                "lock": lock,
            },
            disabled=disabled,
            visible=visible,
        )

    # Resolve type of value to a Literal whenever possible.
    @overload
    def add_gui_select(
        self,
        name: str,
        options: List[TLiteralString],
        initial_value: Optional[TLiteralString] = None,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiSelectHandle[TLiteralString]:
        ...

    @overload
    def add_gui_select(
        self,
        name: str,
        options: List[str],
        initial_value: Optional[str] = None,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiSelectHandle[str]:
        ...

    def add_gui_select(
        self,
        name: str,
        options: List[TLiteralString] | List[str],
        initial_value: Optional[TLiteralString | str] = None,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiSelectHandle[TLiteralString] | GuiSelectHandle[str]:
        """Add a dropdown to the GUI."""
        assert len(options) > 0
        if initial_value is None:
            initial_value = options[0]
        return GuiSelectHandle(
            self._add_gui_impl(
                "/".join(self._gui_folder_labels + [name]),
                initial_value,
                leva_conf={
                    "value": initial_value,
                    "label": name,
                    "options": options,
                },
                disabled=disabled,
                visible=visible,
            )._impl,
            options,  # type: ignore
        )

    def add_gui_slider(
        self,
        name: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: Optional[IntOrFloat],
        initial_value: IntOrFloat,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[IntOrFloat]:
        """Add a slider to the GUI."""
        assert max >= min
        if step is not None:
            assert step <= (max - min)
        assert max >= initial_value >= min

        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={
                "value": initial_value,
                "label": name,
                "min": min,
                "max": max,
                "step": step,
            },
            disabled=disabled,
            visible=visible,
        )

    def add_gui_rgb(
        self,
        name: str,
        initial_value: Tuple[int, int, int],
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[Tuple[int, int, int]]:
        """Add an RGB picker to the GUI."""
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={
                "value": {
                    "r": initial_value[0],
                    "g": initial_value[1],
                    "b": initial_value[2],
                },
                "label": name,
            },
            disabled=disabled,
            visible=visible,
            encoder=lambda rgb: dict(zip("rgb", rgb)),
            decoder=lambda rgb_dict: (rgb_dict["r"], rgb_dict["g"], rgb_dict["b"]),
        )

    def add_gui_rgba(
        self,
        name: str,
        initial_value: Tuple[int, int, int, int],
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiHandle[Tuple[int, int, int, int]]:
        """Add an RGBA picker to the GUI."""
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={
                "value": {
                    "r": initial_value[0],
                    "g": initial_value[1],
                    "b": initial_value[2],
                    "a": initial_value[3],
                },
                "label": name,
            },
            disabled=disabled,
            visible=visible,
            encoder=lambda rgba: dict(zip("rgba", rgba)),
            decoder=lambda rgba_dict: (
                rgba_dict["r"],
                rgba_dict["g"],
                rgba_dict["b"],
                rgba_dict["a"],
            ),
        )

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        color: Tuple[int, int, int]
        | Tuple[float, float, float]
        | onp.ndarray = (80, 120, 255),
        image: Optional[onp.ndarray] = None,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: Optional[int] = None,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
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
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_frame(
        self,
        name: str,
        show_axes: bool = True,
        axes_length: float = 0.5,
        axes_radius: float = 0.025,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
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
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_label(
        self,
        name: str,
        text: str,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
        """Add a 2D label to the scene."""
        self._queue(_messages.LabelMessage(name, text))
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_point_cloud(
        self,
        name: str,
        points: onp.ndarray,
        colors: onp.ndarray,
        point_size: float = 0.1,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
        """Add a point cloud to the scene."""
        self._queue(
            _messages.PointCloudMessage(
                name=name,
                points=points.astype(onp.float32),
                colors=_colors_to_uint8(colors),
                point_size=point_size,
            )
        )
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_mesh(
        self,
        name: str,
        vertices: onp.ndarray,
        faces: onp.ndarray,
        color: Tuple[int, int, int]
        | Tuple[float, float, float]
        | onp.ndarray = (90, 200, 255),
        wireframe: bool = False,
        side: Literal["front", "back", "double"] = "front",
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
        """Add a mesh to the scene."""
        self._queue(
            _messages.MeshMessage(
                name,
                vertices.astype(onp.float32),
                faces.astype(onp.uint32),
                # (255, 255, 255) => 0xffffff, etc
                color=_encode_rgb(color),
                wireframe=wireframe,
                side=side
            )
        )
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

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
    ) -> SceneNodeHandle:
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
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

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

        node_handle = SceneNodeHandle._make(self, name, wxyz, position, visible)
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
        handle_state = self._handle_state_from_gui_name.get(message.name, None)
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

    def _add_gui_impl(
        self,
        name: str,
        initial_value: T,
        leva_conf: dict,
        disabled: bool,
        visible: bool,
        is_button: bool = False,
        encoder: Callable[[T], Any] = lambda x: x,
        decoder: Callable[[Any], T] = lambda x: x,
    ) -> GuiHandle[T]:
        """Private helper for adding a simple GUI element."""

        handle_state = _GuiHandleState(
            name,
            typ=type(initial_value),
            api=self,
            value=initial_value,
            update_timestamp=time.time(),
            folder_labels=self._gui_folder_labels,
            update_cb=[],
            leva_conf=leva_conf,
            is_button=is_button,
            sync_cb=None,
            cleanup_cb=None,
            encoder=encoder,
            decoder=decoder,
            disabled=False,
            visible=True,
        )
        self._handle_state_from_gui_name[name] = handle_state
        handle_state.cleanup_cb = lambda: self._handle_state_from_gui_name.pop(name)

        # For broadcasted GUI handles, we should synchronize all clients.
        # This will be a no-op for client handles.
        if not is_button:

            def sync_other_clients(client_id: ClientId, value: Any) -> None:
                message = _messages.GuiSetValueMessage(
                    name=name, value=handle_state.encoder(value)
                )
                message.excluded_self_client = client_id
                self._queue(message)

            handle_state.sync_cb = sync_other_clients

        self._queue(
            _messages.GuiAddMessage(
                name=name,
                folder_labels=tuple(self._gui_folder_labels),
                leva_conf=leva_conf,
            )
        )
        handle = GuiHandle(handle_state)

        # Set the disabled/visible fields. These will queue messages under-the-hood.
        if disabled:
            handle.disabled = disabled
        if visible:
            handle.visible = visible

        return handle
