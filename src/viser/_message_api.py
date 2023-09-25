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
import queue
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple, TypeVar, Union, cast

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
import trimesh
import trimesh.exchange
import trimesh.visual
from typing_extensions import Literal, ParamSpec, TypeAlias, assert_never

from . import _messages, infra, theme
from ._scene_handles import (
    CameraFrustumHandle,
    ClickEvent,
    FrameHandle,
    GlbHandle,
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


def _hex_from_hls(h: float, l: float, s: float) -> str:
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

    _locked_thread_id: int  # Appeasing mypy 1.5.1, not sure why this is needed.

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
        self._queued_messages: queue.Queue = queue.Queue()
        self._locked_thread_id = -1

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
                h, l, s = colorsys.rgb_to_hls(
                    brand_color[0] / 255.0,
                    brand_color[1] / 255.0,
                    brand_color[2] / 255.0,
                )

                # Automatically generate a 10-color palette.
                min_l = max(l - 0.08, 0.0)
                max_l = min(0.8 + 0.5, 0.9)
                l = max(min_l, min(max_l, l))

                primary_index = 8
                ls = tuple(
                    onp.interp(
                        x=onp.arange(10),
                        xp=(0, primary_index, 9),
                        fp=(max_l, l, min_l),
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

    def add_glb(
        self,
        name,
        glb_data: bytes,
        scale=1.0,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> GlbHandle:
        """Add a general 3D asset via binary glTF (GLB).

        To load glTF files from disk, you can convert to GLB via a library like
        `pygltflib`."""
        self._queue(_messages.GlbMessage(name, glb_data, scale))
        return GlbHandle._make(self, name, wxyz, position, visible)

    def add_spline_catmull_rom(
        self,
        name: str,
        positions: Tuple[Tuple[float, float, float], ...] | onp.ndarray,
        curve_type: Literal["centripetal", "chordal", "catmullrom"] = "centripetal",
        tension: float = 0.5,
        closed: bool = False,
        line_width: float = 1,
        color: RgbTupleOrArray = (20, 20, 20),
    ) -> None:
        """Add spline using Catmull-Rom interpolation."""
        if isinstance(positions, onp.ndarray):
            assert len(positions.shape) == 2 and positions.shape[1] == 3
            positions = tuple(map(tuple, positions))  # type: ignore
        assert len(positions[0]) == 3
        assert isinstance(positions, tuple)
        self._queue(
            _messages.CatmullRomSplineMessage(
                name,
                positions,
                curve_type,
                tension,
                closed,
                line_width,
                _encode_rgb(color),
            )
        )

    def add_spline_cubic_bezier(
        self,
        name: str,
        positions: Tuple[Tuple[float, float, float], ...] | onp.ndarray,
        control_points: Tuple[Tuple[float, float, float], ...] | onp.ndarray,
        line_width: float = 1,
        color: RgbTupleOrArray = (20, 20, 20),
    ) -> None:
        """Add spline using Cubic Bezier interpolation."""

        if isinstance(positions, onp.ndarray):
            assert len(positions.shape) == 2 and positions.shape[1] == 3
            positions = tuple(map(tuple, positions))  # type: ignore
        if isinstance(control_points, onp.ndarray):
            assert len(control_points.shape) == 2 and control_points.shape[1] == 3
            control_points = tuple(map(tuple, control_points))  # type: ignore

        assert isinstance(positions, tuple)
        assert isinstance(control_points, tuple)
        assert len(control_points) == (2 * len(positions) - 2)
        self._queue(
            _messages.CubicBezierSplineMessage(
                name,
                positions,
                control_points,
                line_width,
                _encode_rgb(color),
            )
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
        scale: float = 1.0,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshHandle:
        """Add a trimesh mesh to the scene. Internally calls `seld.add_glb()`."""

        with io.BytesIO() as data_buffer:
            mesh.export(data_buffer, file_type="glb")
            glb_data = data_buffer.getvalue()
            return self.add_glb(
                name,
                glb_data=glb_data,
                scale=scale,
                wxyz=wxyz,
                position=position,
                visible=visible,
            )

    def set_background_image(
        self,
        image: onp.ndarray,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: Optional[int] = None,
        depth: Optional[onp.ndarray] = None,
    ) -> None:
        """Set a background image for the scene, optionally with depth compositing."""
        media_type, base64_data = _encode_image_base64(
            image, format, jpeg_quality=jpeg_quality
        )

        # Encode depth if provided. We use a 3-channel PNG to represent a fixed point
        # depth at each pixel.
        depth_base64data = None
        if depth is not None:
            # Convert to fixed-point.
            # We'll support from 0 -> (2^24 - 1) / 100_000.
            #
            # This translates to a range of [0, 167.77215], with a precision of 1e-5.
            assert len(depth.shape) == 2 or (
                len(depth.shape) == 3 and depth.shape[2] == 1
            ), "Depth should have shape (H,W) or (H,W,1)."
            depth = onp.clip(depth * 100_000, 0, 2**24 - 1).astype(onp.uint32)
            assert depth is not None  # Appease mypy.
            intdepth: onp.ndarray = depth.reshape((*depth.shape[:2], 1)).view(onp.uint8)
            assert intdepth.shape == (*depth.shape[:2], 4)
            with io.BytesIO() as data_buffer:
                iio.imwrite(data_buffer, intdepth[:, :, :3], extension=".png")
                depth_base64data = base64.b64encode(data_buffer.getvalue()).decode(
                    "ascii"
                )

        self._queue(
            _messages.BackgroundImageMessage(
                media_type=media_type,
                base64_rgb=base64_data,
                base64_depth=depth_base64data,
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
        got_lock = self._atomic_lock.acquire(blocking=False)
        if got_lock:
            self._queue_unsafe(message)
            self._atomic_lock.release()
        else:
            # Send when lock is acquirable, while retaining message order.
            # This could be optimized!
            self._queued_messages.put(message)

            def try_again():
                with self._atomic_lock:
                    self._queue_unsafe(self._queued_messages.get())

            threading.Thread(target=try_again).start()

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
            event = ClickEvent(client_id=client_id, target=handle)
            cb(event)  # type: ignore

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

        # New name to make the type checker happy; ViserServer and ClientHandle inherit
        # from both GuiApi and MessageApi. The pattern below is unideal.
        gui_api = self
        assert isinstance(gui_api, GuiApi)

        # Remove the 3D GUI container if it already exists. This will make sure
        # contained GUI elements are removed, preventing potential memory leaks.
        if name in gui_api._handle_from_node_name:
            gui_api._handle_from_node_name[name].remove()

        container_id = _make_unique_id()
        self._queue(
            _messages.Gui3DMessage(
                order=time.time(),
                name=name,
                container_id=container_id,
            )
        )
        node_handle = SceneNodeHandle._make(self, name, wxyz, position)
        return Gui3dContainerHandle(node_handle._impl, gui_api, container_id)
