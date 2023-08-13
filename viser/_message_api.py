# mypy: disable-error-code="misc"
#
# TLiteralString overloads are waiting on PEP 675 support in mypy.
# https://github.com/python/mypy/issues/12554
#
# In the meantime, it works great in Pyright/Pylance!

from __future__ import annotations

import abc
import base64
import colorsys
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, Optional, Tuple, TypeVar, Union, cast

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
import trimesh
import trimesh.visual
from typing_extensions import Literal, ParamSpec, TypeAlias, assert_never

from . import _messages, infra, theme
from ._scene_handles import (
    CameraFrustumHandle,
    FrameHandle,
    Gui3dContainerHandle,
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


def _hex_from_hls(h: float, l: float, s: float) -> str:  # noqa
    """Converts HLS values in [0.0, 1.0] to a hex-formatted string, eg 0xffffff."""
    return "#" + "".join(
        [
            int(min(255, max(0, channel * 255.0)) + 0.5).to_bytes(1, "little").hex()
            for channel in colorsys.hls_to_rgb(h, l, s)
        ]
    )


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


TVector = TypeVar("TVector", bound=tuple)


def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if not isinstance(vector, tuple):
        assert cast(onp.ndarray, vector).shape == (length,)
    return cast(TVector, tuple(map(float, vector)))


class MessageApi(abc.ABC):
    """Interface for all commands we can use to send messages over a websocket connection.

    Should be implemented by both our global server object (for broadcasting) and by
    invidividual clients."""

    def __init__(self, handler: infra.MessageHandler) -> None:
        self._message_handler = handler

        super().__init__()

        self._handle_from_transform_controls_name: Dict[
            str, TransformControlsHandle
        ] = {}
        self._handle_from_node_name: Dict[str, SceneNodeHandle] = {}

        handler.register_handler(
            _messages.TransformControlsUpdateMessage,
            self._handle_transform_controls_updates,
        )
        handler.register_handler(
            _messages.SceneNodeClickedMessage,
            self._handle_click_updates,
        )

        self._atomic_lock = threading.Lock()
        self._locked_thread_id = -1
        self._queue_thread = ThreadPoolExecutor(max_workers=1)

    def configure_theme(
        self,
        *,
        titlebar_content: Optional[theme.TitlebarConfig] = None,
        control_layout: Literal["floating", "collapsible", "fixed"] = "floating",
        dark_mode: bool = False,
        brand_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Configure the viser front-end's visual appearance."""

        colors_cast: Optional[
            Tuple[str, str, str, str, str, str, str, str, str, str]
        ] = None

        if brand_color is not None:
            assert len(brand_color) in (3, 10)
            if len(brand_color) == 3:
                assert all(
                    map(lambda val: isinstance(val, int), brand_color)
                ), "All channels should be integers."

                # RGB => HLS.
                h, primary_l, s = colorsys.rgb_to_hls(
                    brand_color[0] / 255.0,
                    brand_color[1] / 255.0,
                    brand_color[2] / 255.0,
                )

                # Automatically generate a 10-color palette.
                min_l = max(primary_l - 0.08, 0.0)
                max_l = min(0.8 + 0.5, 0.9)
                primary_l = max(min_l, min(max_l, primary_l))

                primary_index = 8
                ls = tuple(
                    onp.interp(
                        x=onp.arange(10),
                        xp=(0, primary_index, 9),
                        fp=(max_l, primary_l, min_l),
                    )
                )
                colors_cast = tuple(_hex_from_hls(h, ls[i], s) for i in range(10))  # type: ignore

        assert colors_cast is None or all(
            [isinstance(val, str) and val.startswith("#") for val in colors_cast]
        ), "All string colors should be in hexadecimal + prefixed with #, eg #ffffff."

        self._queue(
            _messages.ThemeConfigurationMessage(
                titlebar_content=titlebar_content,
                control_layout=control_layout,
                dark_mode=dark_mode,
                colors=colors_cast,
            ),
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
        opacity: Optional[float] = None,
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
                opacity=opacity,
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
                    opacity=None,
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
                    opacity=None,
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

    def _queue(self, message: _messages.Message) -> None:
        """Wrapped method for sending messages safely."""
        # This implementation will retain message ordering because _queue_thread has
        # just 1 worker.
        from .infra._infra import error_print_wrapper

        self._queue_thread.submit(
            error_print_wrapper(lambda: self._queue_blocking(message))
        )

    def _queue_blocking(self, message: _messages.Message) -> None:
        """Wrapped method for sending messages safely. Blocks until ready to send."""
        self._atomic_lock.acquire()
        self._queue_unsafe(message)
        self._atomic_lock.release()

    @abc.abstractmethod
    def _queue_unsafe(self, message: _messages.Message) -> None:
        """Abstract method for sending messages."""
        ...

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

    def add_3d_gui_container(
        self,
        name: str,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
    ) -> Gui3dContainerHandle:
        """Add a 3D gui container to the scene. The returned container handle can be
        used as a context to place GUI elements into the 3D scene."""

        # Avoids circular import.
        from ._gui_api import GuiApi, _make_unique_id

        container_id = _make_unique_id()
        self._queue(
            _messages.Gui3DMessage(
                order=time.time(),
                name=name,
                container_id=container_id,
            )
        )
        assert isinstance(self, MessageApi)
        node_handle = SceneNodeHandle._make(self, name, wxyz, position)
        assert isinstance(self, GuiApi)
        return Gui3dContainerHandle(node_handle._impl, self, container_id)
