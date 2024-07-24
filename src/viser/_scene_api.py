from __future__ import annotations

import base64
import io
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Tuple, TypeVar, Union, cast, get_args

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
from typing_extensions import Literal, ParamSpec, TypeAlias, assert_never

from . import _messages
from . import transforms as tf
from ._scene_handles import (
    BatchedAxesHandle,
    BoneState,
    CameraFrustumHandle,
    FrameHandle,
    GaussianSplatHandle,
    GlbHandle,
    Gui3dContainerHandle,
    ImageHandle,
    LabelHandle,
    MeshHandle,
    MeshSkinnedBoneHandle,
    MeshSkinnedHandle,
    PointCloudHandle,
    SceneNodeHandle,
    SceneNodePointerEvent,
    ScenePointerEvent,
    TransformControlsHandle,
    _SceneNodeHandleState,
    _TransformControlsState,
)

if TYPE_CHECKING:
    import trimesh

    from ._viser import ClientHandle, ViserServer
    from .infra import ClientId


P = ParamSpec("P")


def _colors_to_uint8(colors: onp.ndarray) -> onpt.NDArray[onp.uint8]:
    """Convert intensity values to uint8. We assume the range [0,1] for floats, and
    [0,255] for integers. Accepts any shape."""
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
    jpeg_quality: int | None = None,
) -> tuple[Literal["image/png", "image/jpeg"], str]:
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
                quality=75 if jpeg_quality is None else jpeg_quality,
            )
        else:
            assert_never(format)

        base64_data = base64.b64encode(data_buffer.getvalue()).decode("ascii")

    return media_type, base64_data


TVector = TypeVar("TVector", bound=tuple)


def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if not isinstance(vector, tuple):
        assert cast(onp.ndarray, vector).shape == (
            length,
        ), f"Expected vector of shape {(length,)}, but got {vector.shape} instead"
    return cast(TVector, tuple(map(float, vector)))


class SceneApi:
    """Interface for adding 3D primitives to the scene.

    Used by both our global server object, for sharing the same GUI elements
    with all clients, and by invidividual client handles."""

    def __init__(
        self,
        owner: ViserServer | ClientHandle,  # Who do I belong to?
        thread_executor: ThreadPoolExecutor,
    ) -> None:
        from ._viser import ViserServer

        self._owner = owner
        """Entity that owns this API."""

        self._websock_interface = (
            owner._websock_server
            if isinstance(owner, ViserServer)
            else owner._websock_connection
        )
        """Interface for sending and listening to messages."""

        self.world_axes: FrameHandle = FrameHandle(
            _SceneNodeHandleState(
                "/WorldAxes",
                self,
                wxyz=onp.array([1.0, 0.0, 0.0, 0.0]),
                position=onp.zeros(3),
            )
        )
        """Handle for the world axes, which are created by default."""

        # Hide world axes on initialization.
        if isinstance(owner, ViserServer):
            self.world_axes.visible = False

        self._handle_from_transform_controls_name: dict[
            str, TransformControlsHandle
        ] = {}
        self._handle_from_node_name: dict[str, SceneNodeHandle] = {}

        self._scene_pointer_cb: Callable[[ScenePointerEvent], None] | None = None
        self._scene_pointer_done_cb: Callable[[], None] = lambda: None
        self._scene_pointer_event_type: _messages.ScenePointerEventType | None = None

        self._websock_interface.register_handler(
            _messages.TransformControlsUpdateMessage,
            self._handle_transform_controls_updates,
        )
        self._websock_interface.register_handler(
            _messages.SceneNodeClickMessage,
            self._handle_node_click_updates,
        )
        self._websock_interface.register_handler(
            _messages.ScenePointerMessage,
            self._handle_scene_pointer_updates,
        )

        self._thread_executor = thread_executor

    def set_up_direction(
        self,
        direction: Literal["+x", "+y", "+z", "-x", "-y", "-z"]
        | tuple[float, float, float]
        | onp.ndarray,
    ) -> None:
        """Set the global up direction of the scene. By default we follow +Z-up
        (similar to Blender, 3DS Max, ROS, etc), the most common alternative is
        +Y (OpenGL, Maya, etc).

        Args:
            direction: New up direction. Can either be a string (one of +x, +y,
                +z, -x, -y, -z) or a length-3 direction vector.
        """
        if isinstance(direction, str):
            direction = {
                "+x": (1, 0, 0),
                "+y": (0, 1, 0),
                "+z": (0, 0, 1),
                "-x": (-1, 0, 0),
                "-y": (0, -1, 0),
                "-z": (0, 0, -1),
            }[direction]
        assert not isinstance(direction, str)

        default_three_up = onp.array([0.0, 1.0, 0.0])
        direction = onp.asarray(direction)

        def rotate_between(before: onp.ndarray, after: onp.ndarray) -> tf.SO3:
            assert before.shape == after.shape == (3,)
            before = before / onp.linalg.norm(before)
            after = after / onp.linalg.norm(after)

            angle = onp.arccos(onp.clip(onp.dot(before, after), -1, 1))
            axis = onp.cross(before, after)
            if onp.allclose(axis, onp.zeros(3), rtol=1e-3, atol=1e-5):
                unit_vector = onp.arange(3) == onp.argmin(onp.abs(before))
                axis = onp.cross(before, unit_vector)
            axis = axis / onp.linalg.norm(axis)
            return tf.SO3.exp(angle * axis)

        R_threeworld_world = rotate_between(direction, default_three_up)

        # Rotate the world frame such that:
        #     If we set +Y to up, +X and +Z should face the camera.
        #     If we set +Z to up, +X and +Y should face the camera.
        # In App.tsx, the camera is initialized at [-3, 3, -3] in the threejs
        # coordinate frame.
        desired_fwd = onp.array([-1.0, 0.0, -1.0]) / onp.sqrt(2.0)
        current_fwd = R_threeworld_world @ (onp.ones(3) / onp.sqrt(3.0))
        current_fwd = current_fwd * onp.array([1.0, 0.0, 1.0])
        current_fwd = current_fwd / onp.linalg.norm(current_fwd)
        R_threeworld_world = (
            tf.SO3.from_y_radians(  # Rotate around the null space / up direction.
                onp.arctan2(
                    onp.cross(current_fwd, desired_fwd)[1],
                    onp.dot(current_fwd, desired_fwd),
                ),
            )
            @ R_threeworld_world
        )

        if not onp.any(onp.isnan(R_threeworld_world.wxyz)):
            # Set the orientation of the root node.
            self._websock_interface.queue_message(
                _messages.SetOrientationMessage(
                    "", cast_vector(R_threeworld_world.wxyz, 4)
                )
            )

    def set_global_visibility(self, visible: bool) -> None:
        """Set visibility for all scene nodes. If set to False, all scene nodes
        will be hidden.

        This can be useful when we've called
        :meth:`SceneApi.set_background_image()`, and want to hide everything
        except for the background.

        Args:
            visible: Whether or not all scene nodes should be visible.
        """
        self._websock_interface.queue_message(
            _messages.SetSceneNodeVisibilityMessage("", visible)
        )

    def add_glb(
        self,
        name: str,
        glb_data: bytes,
        scale=1.0,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> GlbHandle:
        """Add a general 3D asset via binary glTF (GLB).

        For glTF files, it's often simpler to use `trimesh.load()` with
        `.add_mesh_trimesh()`. This will call `.add_glb()` under the hood.

        For glTF features not supported by trimesh, glTF to GLB conversion can
        also be done programatically with libraries like `pygltflib`.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
              define a kinematic tree.
            glb_data: A binary payload.
            scale: A scale for resizing the GLB asset.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        self._websock_interface.queue_message(
            _messages.GlbMessage(name, glb_data, scale)
        )
        return GlbHandle._make(self, name, wxyz, position, visible)

    def add_spline_catmull_rom(
        self,
        name: str,
        positions: tuple[tuple[float, float, float], ...] | onp.ndarray,
        curve_type: Literal["centripetal", "chordal", "catmullrom"] = "centripetal",
        tension: float = 0.5,
        closed: bool = False,
        line_width: float = 1,
        color: RgbTupleOrArray = (20, 20, 20),
        segments: int | None = None,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
        """Add a spline to the scene using Catmull-Rom interpolation.

        This method creates a spline based on a set of positions and interpolates
        them using the Catmull-Rom algorithm. This can be used to create smooth curves.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            positions: A tuple of 3D positions (x, y, z) defining the spline's path.
            curve_type: Type of the curve ('centripetal', 'chordal', 'catmullrom').
            tension: Tension of the curve. Affects the tightness of the curve.
            closed: Boolean indicating if the spline is closed (forms a loop).
            line_width: Width of the spline line.
            color: Color of the spline as an RGB tuple.
            segments: Number of segments to divide the spline into.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        if isinstance(positions, onp.ndarray):
            assert len(positions.shape) == 2 and positions.shape[1] == 3
            positions = tuple(map(tuple, positions))  # type: ignore
        assert len(positions[0]) == 3
        assert isinstance(positions, tuple)
        self._websock_interface.queue_message(
            _messages.CatmullRomSplineMessage(
                name,
                positions,
                curve_type,
                tension,
                closed,
                line_width,
                _encode_rgb(color),
                segments=segments,
            )
        )
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_spline_cubic_bezier(
        self,
        name: str,
        positions: tuple[tuple[float, float, float], ...] | onp.ndarray,
        control_points: tuple[tuple[float, float, float], ...] | onp.ndarray,
        line_width: float = 1,
        color: RgbTupleOrArray = (20, 20, 20),
        segments: int | None = None,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
        """Add a spline to the scene using Cubic Bezier interpolation.

        This method allows for the creation of a cubic Bezier spline based on given
        positions and control points. It is useful for creating complex, smooth,
        curving shapes.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            positions: A tuple of 3D positions (x, y, z) defining the spline's key points.
            control_points: A tuple of control points for Bezier curve shaping.
            line_width: Width of the spline line.
            color: Color of the spline as an RGB tuple.
            segments: Number of segments to divide the spline into.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """

        if isinstance(positions, onp.ndarray):
            assert len(positions.shape) == 2 and positions.shape[1] == 3
            positions = tuple(map(tuple, positions))  # type: ignore
        if isinstance(control_points, onp.ndarray):
            assert len(control_points.shape) == 2 and control_points.shape[1] == 3
            control_points = tuple(map(tuple, control_points))  # type: ignore

        assert isinstance(positions, tuple)
        assert isinstance(control_points, tuple)
        assert len(control_points) == (2 * len(positions) - 2)
        self._websock_interface.queue_message(
            _messages.CubicBezierSplineMessage(
                name,
                positions,
                control_points,
                line_width,
                _encode_rgb(color),
                segments=segments,
            )
        )
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        color: RgbTupleOrArray = (20, 20, 20),
        image: onp.ndarray | None = None,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: int | None = None,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> CameraFrustumHandle:
        """Add a camera frustum to the scene for visualization.

        This method adds a frustum representation, typically used to visualize the
        field of view of a camera. It's helpful for understanding the perspective
        and coverage of a camera in the 3D space.

        Like all cameras in the viser Python API, frustums follow the OpenCV [+Z forward,
        +X right, +Y down] convention. fov is vertical in radians; aspect is width over height

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            fov: Field of view of the camera (in radians).
            aspect: Aspect ratio of the camera (width over height).
            scale: Scale factor for the size of the frustum.
            color: Color of the frustum as an RGB tuple.
            image: Optional image to be displayed on the frustum.
            format: Format of the provided image ('png' or 'jpeg').
            jpeg_quality: Quality of the jpeg image (if jpeg format is used).
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """

        if image is not None:
            media_type, base64_data = _encode_image_base64(
                image, format, jpeg_quality=jpeg_quality
            )
        else:
            media_type = None
            base64_data = None

        self._websock_interface.queue_message(
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
        origin_radius: float | None = None,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> FrameHandle:
        """Add a coordinate frame to the scene.

        This method is used for adding a visual representation of a coordinate
        frame, which can help in understanding the orientation and position of
        objects in 3D space.

        For cases where we want to visualize many coordinate frames, like
        trajectories containing thousands or tens of thousands of frames,
        batching and calling `add_batched_axes()` may be a better choice than calling
        `add_frame()` in a loop.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            show_axes: Boolean to indicate whether to show the frame as a set of axes + origin sphere.
            axes_length: Length of each axis.
            axes_radius: Radius of each axis.
            origin_radius: Radius of the origin sphere. If not set, defaults to `2 * axes_radius`.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        if origin_radius is None:
            origin_radius = axes_radius * 2
        self._websock_interface.queue_message(
            _messages.FrameMessage(
                name=name,
                show_axes=show_axes,
                axes_length=axes_length,
                axes_radius=axes_radius,
                origin_radius=origin_radius,
            )
        )
        return FrameHandle._make(self, name, wxyz, position, visible)

    def add_batched_axes(
        self,
        name: str,
        batched_wxyzs: tuple[tuple[float, float, float, float], ...] | onp.ndarray,
        batched_positions: tuple[tuple[float, float, float], ...] | onp.ndarray,
        axes_length: float = 0.5,
        axes_radius: float = 0.025,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> BatchedAxesHandle:
        """Visualize batched sets of coordinate frame axes.

        The functionality of `add_batched_axes()` overlaps significantly with
        `add_frame()` when `show_axes=True`. The primary difference is that
        `add_batched_axes()` supports multiple axes via the `wxyzs_batched`
        (shape Nx4) and `positions_batched` (shape Nx3) arguments.

        Axes that are batched and rendered via a single call to
        `add_batched_axes()` are instanced on the client; this will be much
        faster to render than `add_frame()` called in a loop.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            batched_wxyzs: Float array of shape (N,4).
            batched_positions: Float array of shape (N,3).
            axes_length: Length of each axis.
            axes_radius: Radius of each axis.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
                This will be applied to all axes.
            position: Translation to parent frame from local frame (t_pl).
                This will be applied to all axes.
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        batched_wxyzs = onp.asarray(batched_wxyzs)
        batched_positions = onp.asarray(batched_positions)

        num_axes = batched_wxyzs.shape[0]
        assert batched_wxyzs.shape == (num_axes, 4)
        assert batched_positions.shape == (num_axes, 3)
        self._websock_interface.queue_message(
            _messages.BatchedAxesMessage(
                name=name,
                wxyzs_batched=batched_wxyzs.astype(onp.float32),
                positions_batched=batched_positions.astype(onp.float32),
                axes_length=axes_length,
                axes_radius=axes_radius,
            )
        )
        return BatchedAxesHandle._make(self, name, wxyz, position, visible)

    def add_grid(
        self,
        name: str,
        width: float = 10.0,
        height: float = 10.0,
        width_segments: int = 10,
        height_segments: int = 10,
        plane: Literal["xz", "xy", "yx", "yz", "zx", "zy"] = "xy",
        cell_color: RgbTupleOrArray = (200, 200, 200),
        cell_thickness: float = 1.0,
        cell_size: float = 0.5,
        section_color: RgbTupleOrArray = (140, 140, 140),
        section_thickness: float = 1.0,
        section_size: float = 1.0,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> SceneNodeHandle:
        """Add a 2D grid to the scene.

        This can be useful as a size, orientation, or ground plane reference.

        Args:
            name: Name of the grid.
            width: Width of the grid.
            height: Height of the grid.
            width_segments: Number of segments along the width.
            height_segments: Number of segments along the height.
            plane: The plane in which the grid is oriented (e.g., 'xy', 'yz').
            cell_color: Color of the grid cells as an RGB tuple.
            cell_thickness: Thickness of the grid lines.
            cell_size: Size of each cell in the grid.
            section_color: Color of the grid sections as an RGB tuple.
            section_thickness: Thickness of the section lines.
            section_size: Size of each section in the grid.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        self._websock_interface.queue_message(
            _messages.GridMessage(
                name=name,
                width=width,
                height=height,
                width_segments=width_segments,
                height_segments=height_segments,
                plane=plane,
                cell_color=_encode_rgb(cell_color),
                cell_thickness=cell_thickness,
                cell_size=cell_size,
                section_color=_encode_rgb(section_color),
                section_thickness=section_thickness,
                section_size=section_size,
            )
        )
        return SceneNodeHandle._make(self, name, wxyz, position, visible)

    def add_label(
        self,
        name: str,
        text: str,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> LabelHandle:
        """Add a 2D label to the scene.

        This method creates a text label in the 3D scene, which can be used to annotate
        or provide information about specific points or objects.

        Args:
            name: Name of the label.
            text: Text content of the label.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        self._websock_interface.queue_message(_messages.LabelMessage(name, text))
        return LabelHandle._make(self, name, wxyz, position, visible=visible)

    def add_point_cloud(
        self,
        name: str,
        points: onp.ndarray,
        colors: onp.ndarray | tuple[float, float, float],
        point_size: float = 0.1,
        point_shape: Literal[
            "square", "diamond", "circle", "rounded", "sparkle"
        ] = "square",
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> PointCloudHandle:
        """Add a point cloud to the scene.

        Args:
            name: Name of scene node. Determines location in kinematic tree.
            points: Location of points. Should have shape (N, 3).
            colors: Colors of points. Should have shape (N, 3) or (3,).
            point_size: Size of each point.
            point_shape: Shape to draw each point.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        colors_cast = _colors_to_uint8(onp.asarray(colors))
        assert (
            len(points.shape) == 2 and points.shape[-1] == 3
        ), "Shape of points should be (N, 3)."
        assert colors_cast.shape in {
            points.shape,
            (3,),
        }, "Shape of colors should be (N, 3) or (3,)."

        if colors_cast.shape == (3,):
            colors_cast = onp.tile(colors_cast[None, :], reps=(points.shape[0], 1))

        self._websock_interface.queue_message(
            _messages.PointCloudMessage(
                name=name,
                points=points.astype(onp.float32),
                colors=colors_cast,
                point_size=point_size,
                point_ball_norm={
                    "square": float("inf"),
                    "diamond": 1.0,
                    "circle": 2.0,
                    "rounded": 3.0,
                    "sparkle": 0.6,
                }[point_shape],
            )
        )
        return PointCloudHandle._make(self, name, wxyz, position, visible)

    def add_mesh_skinned(
        self,
        name: str,
        vertices: onp.ndarray,
        faces: onp.ndarray,
        bone_wxyzs: tuple[tuple[float, float, float, float], ...] | onp.ndarray,
        bone_positions: tuple[tuple[float, float, float], ...] | onp.ndarray,
        skin_weights: onp.ndarray,
        color: RgbTupleOrArray = (90, 200, 255),
        wireframe: bool = False,
        opacity: float | None = None,
        material: Literal["standard", "toon3", "toon5"] = "standard",
        flat_shading: bool = False,
        side: Literal["front", "back", "double"] = "front",
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshSkinnedHandle:
        """Add a skinned mesh to the scene, which we can deform using a set of
        bone transformations.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            vertices: A numpy array of vertex positions. Should have shape (V, 3).
            faces: A numpy array of faces, where each face is represented by indices of
                vertices. Should have shape (F,)
            bone_wxyzs: Nested tuple or array of initial bone orientations.
            bone_positions: Nested tuple or array of initial bone positions.
            skin_weights: A numpy array of skin weights. Should have shape (V, B) where B
                is the number of bones. Only the top 4 bone weights for each
                vertex will be used.
            color: Color of the mesh as an RGB tuple.
            wireframe: Boolean indicating if the mesh should be rendered as a wireframe.
            opacity: Opacity of the mesh. None means opaque.
            material: Material type of the mesh ('standard', 'toon3', 'toon5').
                This argument is ignored when wireframe=True.
            flat_shading: Whether to do flat shading. This argument is ignored
                when wireframe=True.
            side: Side of the surface to render ('front', 'back', 'double').
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation from parent frame to local frame (t_pl).
            visible: Whether or not this mesh is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        if wireframe and material != "standard":
            warnings.warn(
                f"Invalid combination of {wireframe=} and {material=}. Material argument will be ignored.",
                stacklevel=2,
            )
        if wireframe and flat_shading:
            warnings.warn(
                f"Invalid combination of {wireframe=} and {flat_shading=}. Flat shading argument will be ignored.",
                stacklevel=2,
            )

        num_bones = len(bone_wxyzs)
        assert skin_weights.shape == (vertices.shape[0], num_bones)

        # Take the four biggest indices.
        top4_skin_indices = onp.argsort(skin_weights, axis=-1)[:, -4:]
        top4_skin_weights = skin_weights[
            onp.arange(vertices.shape[0])[:, None], top4_skin_indices
        ]
        assert (
            top4_skin_weights.shape == top4_skin_indices.shape == (vertices.shape[0], 4)
        )

        bone_wxyzs = onp.asarray(bone_wxyzs)
        bone_positions = onp.asarray(bone_positions)
        assert bone_wxyzs.shape == (num_bones, 4)
        assert bone_positions.shape == (num_bones, 3)
        self._websock_interface.queue_message(
            _messages.SkinnedMeshMessage(
                name,
                vertices.astype(onp.float32),
                faces.astype(onp.uint32),
                # (255, 255, 255) => 0xffffff, etc
                color=_encode_rgb(color),
                vertex_colors=None,
                wireframe=wireframe,
                opacity=opacity,
                flat_shading=flat_shading,
                side=side,
                material=material,
                bone_wxyzs=tuple(
                    (
                        float(wxyz[0]),
                        float(wxyz[1]),
                        float(wxyz[2]),
                        float(wxyz[3]),
                    )
                    for wxyz in bone_wxyzs.astype(onp.float32)
                ),
                bone_positions=tuple(
                    (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    for xyz in bone_positions.astype(onp.float32)
                ),
                skin_indices=top4_skin_indices.astype(onp.uint16),
                skin_weights=top4_skin_weights.astype(onp.float32),
            )
        )
        handle = MeshHandle._make(self, name, wxyz, position, visible)
        return MeshSkinnedHandle(
            handle._impl,
            bones=tuple(
                MeshSkinnedBoneHandle(
                    _impl=BoneState(
                        name=name,
                        websock_interface=self._websock_interface,
                        bone_index=i,
                        wxyz=bone_wxyzs[i],
                        position=bone_positions[i],
                    )
                )
                for i in range(num_bones)
            ),
        )

    def add_mesh_simple(
        self,
        name: str,
        vertices: onp.ndarray,
        faces: onp.ndarray,
        color: RgbTupleOrArray = (90, 200, 255),
        wireframe: bool = False,
        opacity: float | None = None,
        material: Literal["standard", "toon3", "toon5"] = "standard",
        flat_shading: bool = False,
        side: Literal["front", "back", "double"] = "front",
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshHandle:
        """Add a mesh to the scene.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            vertices: A numpy array of vertex positions. Should have shape (V, 3).
            faces: A numpy array of faces, where each face is represented by indices of
                vertices. Should have shape (F,)
            color: Color of the mesh as an RGB tuple.
            wireframe: Boolean indicating if the mesh should be rendered as a wireframe.
            opacity: Opacity of the mesh. None means opaque.
            material: Material type of the mesh ('standard', 'toon3', 'toon5').
                This argument is ignored when wireframe=True.
            flat_shading: Whether to do flat shading. This argument is ignored
                when wireframe=True.
            side: Side of the surface to render ('front', 'back', 'double').
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation from parent frame to local frame (t_pl).
            visible: Whether or not this mesh is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        if wireframe and material != "standard":
            warnings.warn(
                f"Invalid combination of {wireframe=} and {material=}. Material argument will be ignored.",
                stacklevel=2,
            )
        if wireframe and flat_shading:
            warnings.warn(
                f"Invalid combination of {wireframe=} and {flat_shading=}. Flat shading argument will be ignored.",
                stacklevel=2,
            )

        self._websock_interface.queue_message(
            _messages.MeshMessage(
                name,
                vertices.astype(onp.float32),
                faces.astype(onp.uint32),
                # (255, 255, 255) => 0xffffff, etc
                color=_encode_rgb(color),
                vertex_colors=None,
                wireframe=wireframe,
                opacity=opacity,
                flat_shading=flat_shading,
                side=side,
                material=material,
            )
        )
        return MeshHandle._make(self, name, wxyz, position, visible)

    def add_mesh_trimesh(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        scale: float = 1.0,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> GlbHandle:
        """Add a trimesh mesh to the scene. Internally calls `self.add_glb()`.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
              define a kinematic tree.
            mesh: A trimesh mesh object.
            scale: A scale for resizing the mesh.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node.
        """

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

    def _add_gaussian_splats(
        self,
        name: str,
        centers: onp.ndarray,
        covariances: onp.ndarray,
        rgbs: onp.ndarray,
        opacities: onp.ndarray,
        wxyz: Tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: Tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> GaussianSplatHandle:
        """Add a model to render using Gaussian Splatting.

        Limitations: (i) does not yet support spherical harmonics, and (ii) our
        shader supports a limited nmber of splat objects (currently 32).

        **Work-in-progress.** This feature is experimental and still under
        development. It may be changed or removed.

        Arguments:
            name: Scene node name.
            centers: Centers of Gaussians. (N, 3).
            covariances: Second moment for each Gaussian. (N, 3, 3).
            rgbs: Color for each Gaussian. (N, 3).
            opacities: Opacity for each Gaussian. (N, 1).
            wxyz: R_parent_local transformation.
            position: t_parent_local transformation.
            visibile: Initial visibility of scene node.

        Returns:
            Scene node handle.
        """
        num_gaussians = centers.shape[0]
        assert centers.shape == (num_gaussians, 3)
        assert rgbs.shape == (num_gaussians, 3)
        assert opacities.shape == (num_gaussians, 1)
        assert covariances.shape == (num_gaussians, 3, 3)

        # Get cholesky factor of covariance.
        cov_cholesky_triu = (
            onp.linalg.cholesky(covariances.astype(onp.float64) + onp.ones(3) * 1e-7)
            .swapaxes(-1, -2)  # tril => triu
            .reshape((-1, 9))[:, onp.array([0, 1, 2, 4, 5, 8])]
        )
        buffer = onp.concatenate(
            [
                # First texelFetch.
                centers.astype(onp.float32).view(onp.uint8),
                onp.zeros((num_gaussians, 4), dtype=onp.uint8),
                # Second texelFetch.
                cov_cholesky_triu.astype(onp.float16).copy().view(onp.uint8),
                _colors_to_uint8(rgbs),
                _colors_to_uint8(opacities),
            ],
            axis=-1,
        ).view(onp.uint32)
        assert buffer.shape == (num_gaussians, 8)

        self._websock_interface.queue_message(
            _messages.GaussianSplatsMessage(
                name=name,
                buffer=buffer,
            )
        )
        node_handle = GaussianSplatHandle._make(self, name, wxyz, position, visible)
        return node_handle

    def add_box(
        self,
        name: str,
        color: RgbTupleOrArray,
        dimensions: tuple[float, float, float] | onp.ndarray = (1.0, 1.0, 1.0),
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshHandle:
        """Add a box to the scene.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            color: Color of the box as an RGB tuple.
            dimensions: Dimensions of the box (x, y, z).
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation from parent frame to local frame (t_pl).
            visible: Whether or not this box is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        import trimesh.creation

        mesh = trimesh.creation.box(dimensions)

        return self.add_mesh_simple(
            name=name,
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=color,
            flat_shading=True,
            position=position,
            wxyz=wxyz,
            visible=visible,
        )

    def add_icosphere(
        self,
        name: str,
        radius: float,
        color: RgbTupleOrArray,
        subdivisions: int = 3,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> MeshHandle:
        """Add an icosphere to the scene.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            radius: Radius of the icosphere.
            color: Color of the icosphere as an RGB tuple.
            subdivisions: Number of subdivisions to use when creating the icosphere.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation from parent frame to local frame (t_pl).
            visible: Whether or not this icosphere is initially visible.

        Returns:
            Handle for manipulating scene node.
        """
        import trimesh.creation

        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

        # We use add_mesh_simple() because it lets us do smooth shading;
        # add_mesh_trimesh() currently does not.
        return self.add_mesh_simple(
            name=name,
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=color,
            flat_shading=False,
            position=position,
            wxyz=wxyz,
            visible=visible,
        )

    def set_background_image(
        self,
        image: onp.ndarray,
        format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: int | None = None,
        depth: onp.ndarray | None = None,
    ) -> None:
        """Set a background image for the scene, optionally with depth compositing.

        Args:
            image: The image to set as the background. Should have shape (H, W, 3).
            format: Format to transport and display the image using ('png' or 'jpeg').
            jpeg_quality: Quality of the jpeg image (if jpeg format is used).
            depth: Optional depth image to use to composite background with scene elements.
        """
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

        self._websock_interface.queue_message(
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
        jpeg_quality: int | None = None,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> ImageHandle:
        """Add a 2D image to the scene.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            image: A numpy array representing the image.
            render_width: Width at which the image should be rendered in the scene.
            render_height: Height at which the image should be rendered in the scene.
            format: Format to transport and display the image using ('png' or 'jpeg').
            jpeg_quality: Quality of the jpeg image (if jpeg format is used).
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation from parent frame to local frame (t_pl).
            visible: Whether or not this image is initially visible.

        Returns:
            Handle for manipulating scene node.
        """

        media_type, base64_data = _encode_image_base64(
            image, format, jpeg_quality=jpeg_quality
        )
        self._websock_interface.queue_message(
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
        active_axes: tuple[bool, bool, bool] = (True, True, True),
        disable_axes: bool = False,
        disable_sliders: bool = False,
        disable_rotations: bool = False,
        translation_limits: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ] = ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        rotation_limits: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ] = ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        depth_test: bool = True,
        opacity: float = 1.0,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> TransformControlsHandle:
        """Add a transform gizmo for interacting with the scene.

        This method adds a transform control (gizmo) to the scene, allowing for interactive
        manipulation of objects in terms of their position, rotation, and scale.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            scale: Scale of the transform controls.
            line_width: Width of the lines used in the gizmo.
            fixed: Boolean indicating if the gizmo should be fixed in position.
            auto_transform: Whether the transform should be applied automatically.
            active_axes: tuple of booleans indicating active axes.
            disable_axes: Boolean to disable axes interaction.
            disable_sliders: Boolean to disable slider interaction.
            disable_rotations: Boolean to disable rotation interaction.
            translation_limits: Limits for translation.
            rotation_limits: Limits for rotation.
            depth_test: Boolean indicating if depth testing should be used when rendering.
            opacity: Opacity of the gizmo.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation from parent frame to local frame (t_pl).
            visible: Whether or not this gizmo is initially visible.

        Returns:
            Handle for manipulating (and reading state of) scene node.
        """
        self._websock_interface.queue_message(
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
            self._websock_interface.queue_message(message_orientation)

            message_position = _messages.SetPositionMessage(
                name=name,
                position=tuple(map(float, state._impl.position)),  # type: ignore
            )
            message_position.excluded_self_client = client_id
            self._websock_interface.queue_message(message_position)

        node_handle = SceneNodeHandle._make(self, name, wxyz, position, visible)
        state_aux = _TransformControlsState(
            last_updated=time.time(),
            update_cb=[],
            sync_cb=sync_cb,
        )
        handle = TransformControlsHandle(node_handle._impl, state_aux)
        self._handle_from_transform_controls_name[name] = handle
        return handle

    def reset(self) -> None:
        """Reset the scene."""
        self._websock_interface.queue_message(_messages.ResetSceneMessage())

    def _get_client_handle(self, client_id: ClientId) -> ClientHandle:
        """Private helper for getting a client handle from its ID."""
        # Avoid circular imports.
        from ._viser import ViserServer

        # Implementation-wise, note that MessageApi is never directly instantiated.
        # Instead, it serves as a mixin/base class for either ViserServer, which
        # maintains a registry of connected clients, or ClientHandle, which should
        # only ever be dealing with its own client_id.
        if isinstance(self._owner, ViserServer):
            # TODO: there's a potential race condition here when the client disconnects.
            # This probably applies to multiple other parts of the code, we should
            # revisit all of the cases where we index into connected_clients.
            return self._owner._connected_clients[client_id]
        else:
            assert client_id == self._owner.client_id
            return self._owner

    def _handle_transform_controls_updates(
        self, client_id: ClientId, message: _messages.TransformControlsUpdateMessage
    ) -> None:
        """Callback for handling transform gizmo messages."""
        handle = self._handle_from_transform_controls_name.get(message.name, None)
        if handle is None:
            return

        # Update state.
        wxyz = onp.array(message.wxyz)
        position = onp.array(message.position)
        with self._owner.atomic():
            handle._impl.wxyz = wxyz
            handle._impl.position = position
            handle._impl_aux.last_updated = time.time()

        # Trigger callbacks.
        for cb in handle._impl_aux.update_cb:
            cb(handle)
        if handle._impl_aux.sync_cb is not None:
            handle._impl_aux.sync_cb(client_id, handle)

    def _handle_node_click_updates(
        self, client_id: ClientId, message: _messages.SceneNodeClickMessage
    ) -> None:
        """Callback for handling click messages."""
        handle = self._handle_from_node_name.get(message.name, None)
        if handle is None or handle._impl.click_cb is None:
            return
        for cb in handle._impl.click_cb:
            event = SceneNodePointerEvent(
                client=self._get_client_handle(client_id),
                client_id=client_id,
                event="click",
                target=handle,
                ray_origin=message.ray_origin,
                ray_direction=message.ray_direction,
            )
            cb(event)  # type: ignore

    def _handle_scene_pointer_updates(
        self, client_id: ClientId, message: _messages.ScenePointerMessage
    ):
        """Callback for handling click messages."""
        event = ScenePointerEvent(
            client=self._get_client_handle(client_id),
            client_id=client_id,
            event_type=message.event_type,
            ray_origin=message.ray_origin,
            ray_direction=message.ray_direction,
            screen_pos=message.screen_pos,
        )
        # Call the callback if it exists, and the after-run callback.
        if self._scene_pointer_cb is None:
            return
        self._scene_pointer_cb(event)

    def on_pointer_event(
        self, event_type: Literal["click", "rect-select"]
    ) -> Callable[
        [Callable[[ScenePointerEvent], None]], Callable[[ScenePointerEvent], None]
    ]:
        """Add a callback for scene pointer events.

        Args:
            event_type: event to listen to.
        """
        # Ensure the event type is valid.
        assert event_type in get_args(_messages.ScenePointerEventType)

        from ._viser import ClientHandle, ViserServer

        def cleanup_previous_event(target: ViserServer | ClientHandle):
            # If the server or client does not have a scene pointer callback, return.
            if target.scene._scene_pointer_cb is None:
                return

            # Remove callback.
            target.scene.remove_pointer_callback()

        def decorator(
            func: Callable[[ScenePointerEvent], None],
        ) -> Callable[[ScenePointerEvent], None]:
            # Check if another scene pointer event was previously registered.
            # If so, we need to clear the previous event and register the new one.
            cleanup_previous_event(self._owner)

            # If called on the server handle, remove all clients' callbacks.
            if isinstance(self._owner, ViserServer):
                for client in self._owner.get_clients().values():
                    cleanup_previous_event(client)

            # If called on the client handle, and server handle has a callback, remove the server's callback.
            # (If the server has a callback, none of the clients should have callbacks.)
            elif isinstance(self._owner, ClientHandle):
                server = self._owner._viser_server
                cleanup_previous_event(server)

            self._scene_pointer_cb = func
            self._scene_pointer_event_type = event_type

            self._websock_interface.queue_message(
                _messages.ScenePointerEnableMessage(enable=True, event_type=event_type)
            )
            return func

        return decorator

    def on_pointer_callback_removed(
        self,
        func: Callable[[], None],
    ) -> Callable[[], None]:
        """Add a callback to run automatically when the callback for a scene
        pointer event is removed. This will be triggered exactly once, either
        manually (via :meth:`remove_pointer_callback()`) or automatically (if
        the scene pointer event is overridden with another call to
        :meth:`on_pointer_event()`).

        Args:
            func: Callback for when scene pointer events are removed.
        """
        self._scene_pointer_done_cb = func
        return func

    def remove_pointer_callback(
        self,
    ) -> None:
        """Remove the currently attached scene pointer event. This will trigger
        any callback attached to `.on_scene_pointer_removed()`."""

        if self._scene_pointer_cb is None:
            warnings.warn(
                "No scene pointer callback exists for this server/client, ignoring.",
                stacklevel=2,
            )
            return

        # Notify client that the listener has been removed.
        event_type = self._scene_pointer_event_type
        assert event_type is not None
        self._websock_interface.queue_message(
            _messages.ScenePointerEnableMessage(enable=False, event_type=event_type)
        )
        self._owner.flush()

        # Run cleanup callback.
        self._scene_pointer_done_cb()

        # Reset the callback and event type, on the python side.
        self._scene_pointer_cb = None
        self._scene_pointer_done_cb = lambda: None
        self._scene_pointer_event_type = None

    def add_3d_gui_container(
        self,
        name: str,
        wxyz: tuple[float, float, float, float] | onp.ndarray = (1.0, 0.0, 0.0, 0.0),
        position: tuple[float, float, float] | onp.ndarray = (0.0, 0.0, 0.0),
        visible: bool = True,
    ) -> Gui3dContainerHandle:
        """Add a 3D gui container to the scene. The returned container handle can be
        used as a context to place GUI elements into the 3D scene.

        Args:
            name: A scene tree name. Names in the format of /parent/child can be used to
                define a kinematic tree.
            wxyz: Quaternion rotation to parent frame from local frame (R_pl).
            position: Translation to parent frame from local frame (t_pl).
            visible: Whether or not this scene node is initially visible.

        Returns:
            Handle for manipulating scene node. Can be used as a context to place GUI
            elements inside of the container.
        """

        # Avoids circular import.
        from ._gui_api import _make_unique_id

        # New name to make the type checker happy; ViserServer and ClientHandle inherit
        # from both GuiApi and MessageApi. The pattern below is unideal.
        gui_api = self._owner.gui

        # Remove the 3D GUI container if it already exists. This will make sure
        # contained GUI elements are removed, preventing potential memory leaks.
        if name in self._handle_from_node_name:
            self._handle_from_node_name[name].remove()

        container_id = _make_unique_id()
        self._websock_interface.queue_message(
            _messages.Gui3DMessage(
                order=time.time(),
                name=name,
                container_id=container_id,
            )
        )
        node_handle = SceneNodeHandle._make(self, name, wxyz, position, visible=visible)
        return Gui3dContainerHandle(node_handle._impl, gui_api, container_id)
