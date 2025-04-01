from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Coroutine
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as onpt
from typing_extensions import Self, get_type_hints, override

from . import _messages
from .infra._infra import WebsockClientConnection, WebsockServer

if TYPE_CHECKING:
    from ._gui_api import GuiApi
    from ._scene_api import SceneApi
    from ._viser import ClientHandle
    from .infra import ClientId


def colors_to_uint8(colors: np.ndarray) -> onpt.NDArray[np.uint8]:
    """Convert intensity values to uint8. We assume the range [0,1] for floats, and
    [0,255] for integers. Accepts any shape."""
    if colors.dtype != np.uint8:
        if np.issubdtype(colors.dtype, np.floating):
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        if np.issubdtype(colors.dtype, np.integer):
            colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


class _HasCastArrayDtypes:
    def _cast_array_dtypes(
        self, prop_hints: Dict[str, Any], prop_name: str, value: np.ndarray
    ) -> np.ndarray:
        """Helper used by __setattr__, which casts assigned values to the
        correct type."""
        hint = prop_hints[prop_name]
        if hint == onpt.NDArray[np.float32]:
            value = value.astype(np.float32)
        elif hint == onpt.NDArray[np.float16]:
            value = value.astype(np.float16)
        elif hint == onpt.NDArray[np.uint8] and "color" in prop_name:
            value = colors_to_uint8(value)
        return value


class _OverridableScenePropApi(_HasCastArrayDtypes):
    """Mixin that allows reading/assigning properties defined in each scene
    node message."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_impl":
            return object.__setattr__(self, name, value)

        handle = cast(SceneNodeHandle, self)
        # Get the value of the T TypeVar.
        if name in self._prop_hints:
            value = self._cast_array_dtypes(self._prop_hints, name, value)

            current_value = getattr(handle._impl.props, name)

            # Do nothing if the value hasn't changed.
            if isinstance(current_value, np.ndarray):
                if current_value.data == value.data:
                    return
            elif current_value == value:
                return

            # Update the value.
            if isinstance(value, np.ndarray):
                assert value.dtype == current_value.dtype
                if value.shape == current_value.shape:
                    current_value[:] = value
                else:
                    setattr(handle._impl.props, name, value.copy())
            else:
                # Non-array properties should be immutable, so no need to copy.
                setattr(handle._impl.props, name, value)

            handle._impl.api._websock_interface.queue_message(
                _messages.SceneNodeUpdateMessage(handle.name, {name: value})
            )
        else:
            return object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._prop_hints:
            return getattr(self._impl.props, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    @cached_property
    def _prop_hints(self) -> Dict[str, Any]:
        return get_type_hints(type(self._impl.props))


@dataclasses.dataclass(frozen=True)
class ScenePointerEvent:
    """Event passed to pointer callbacks for the scene (currently only clicks)."""

    client: ClientHandle
    """Client that triggered this event."""
    client_id: int
    """ID of client that triggered this event."""
    event_type: _messages.ScenePointerEventType
    """Type of event that was triggered. Currently we only support clicks and box selections."""
    ray_origin: tuple[float, float, float] | None
    """Origin of 3D ray corresponding to this click, in world coordinates."""
    ray_direction: tuple[float, float, float] | None
    """Direction of 3D ray corresponding to this click, in world coordinates."""
    screen_pos: tuple[tuple[float, float], ...]
    """Screen position of the click on the screen (OpenCV image coordinates, 0 to 1).
    (0, 0) is the upper-left corner, (1, 1) is the bottom-right corner.
    For a box selection, this includes the min- and max- corners of the box."""

    @property
    def event(self):
        """Deprecated. Use `event_type` instead."""
        return self.event_type


TSceneNodeHandle = TypeVar("TSceneNodeHandle", bound="SceneNodeHandle")


@dataclasses.dataclass
class _SceneNodeHandleState:
    name: str
    props: Any  # _messages.*Prop object.
    """Message containing properties of this scene node that are sent to the
    client."""
    api: SceneApi
    wxyz: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    position: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    visible: bool = True
    click_cb: list[
        Callable[[SceneNodePointerEvent[_ClickableSceneNodeHandle]], None | Coroutine]
    ] = dataclasses.field(default_factory=list)
    removed: bool = False


class _SceneNodeMessage(Protocol):
    name: str
    props: Any


class SceneNodeHandle:
    """Handle base class for interacting with scene nodes."""

    def __init__(self, impl: _SceneNodeHandleState) -> None:
        self._impl = impl

    @property
    def name(self) -> str:
        """Read-only name of the scene node."""
        return self._impl.name

    @classmethod
    def _make(
        cls: type[TSceneNodeHandle],
        api: SceneApi,
        message: _SceneNodeMessage,
        name: str,
        wxyz: tuple[float, float, float, float] | np.ndarray,
        position: tuple[float, float, float] | np.ndarray,
        visible: bool,
    ) -> TSceneNodeHandle:
        """Create scene node: send state to client(s) and set up
        server-side state."""
        # Send message.
        assert isinstance(message, _messages.Message)
        api._websock_interface.queue_message(message)

        out = cls(_SceneNodeHandleState(name, copy.deepcopy(message.props), api))
        api._handle_from_node_name[name] = out

        out.wxyz = wxyz
        out.position = position

        # Toggle visibility to make sure we send a
        # SetSceneNodeVisibilityMessage to the client.
        out._impl.visible = not visible
        out.visible = visible
        return out

    @property
    def wxyz(self) -> np.ndarray:
        """Orientation of the scene node. This is the quaternion representation of the R
        in `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: tuple[float, float, float, float] | np.ndarray) -> None:
        from ._scene_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        wxyz_array = np.asarray(wxyz)
        if np.allclose(wxyz_cast, self._impl.wxyz):
            return
        self._impl.wxyz[:] = wxyz_array
        self._impl.api._websock_interface.queue_message(
            _messages.SetOrientationMessage(self._impl.name, wxyz_cast)
        )

    @property
    def position(self) -> np.ndarray:
        """Position of the scene node. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: tuple[float, float, float] | np.ndarray) -> None:
        from ._scene_api import cast_vector

        position_cast = cast_vector(position, 3)
        position_array = np.asarray(position)
        if np.allclose(position_array, self._impl.position):
            return
        self._impl.position[:] = position_array
        self._impl.api._websock_interface.queue_message(
            _messages.SetPositionMessage(self._impl.name, position_cast)
        )

    @property
    def visible(self) -> bool:
        """Whether the scene node is visible or not. Synchronized to clients automatically when assigned."""
        return self._impl.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self._impl.visible:
            return
        self._impl.api._websock_interface.queue_message(
            _messages.SetSceneNodeVisibilityMessage(self._impl.name, visible)
        )
        self._impl.visible = visible

    def remove(self) -> None:
        """Remove the node from the scene."""
        # Warn if already removed.
        if self._impl.removed:
            warnings.warn(f"Attempted to remove already removed node: {self.name}")
            return

        self._impl.removed = True
        self._impl.api._handle_from_node_name.pop(self._impl.name)
        self._impl.api._websock_interface.queue_message(
            _messages.RemoveSceneNodeMessage(self._impl.name)
        )


@dataclasses.dataclass(frozen=True)
class SceneNodePointerEvent(Generic[TSceneNodeHandle]):
    """Event passed to pointer callbacks for scene nodes (currently only clicks)."""

    client: ClientHandle
    """Client that triggered this event."""
    client_id: int
    """ID of client that triggered this event."""
    event: Literal["click"]
    """Type of event that was triggered. Currently we only support clicks."""
    target: TSceneNodeHandle
    """Scene node that was clicked."""
    ray_origin: tuple[float, float, float]
    """Origin of 3D ray corresponding to this click, in world coordinates."""
    ray_direction: tuple[float, float, float]
    """Direction of 3D ray corresponding to this click, in world coordinates."""
    screen_pos: tuple[float, float]
    """Screen position of the click on the screen (OpenCV image coordinates, 0 to 1).
    (0, 0) is the upper-left corner, (1, 1) is the bottom-right corner."""
    instance_index: int | None
    """Instance ID of the clicked object, if applicable. Currently this is `None` for all objects except for the output of :meth:`SceneApi.add_batched_axes()`."""


NoneOrCoroutine = TypeVar("NoneOrCoroutine", None, Coroutine)


class _ClickableSceneNodeHandle(SceneNodeHandle):
    def on_click(
        self: Self,
        func: Callable[[SceneNodePointerEvent[Self]], NoneOrCoroutine],
    ) -> Callable[[SceneNodePointerEvent[Self]], NoneOrCoroutine]:
        """Attach a callback for when a scene node is clicked.

        The callback can be either a standard function or an async function:
        - Standard functions (def) will be executed in a threadpool.
        - Async functions (async def) will be executed in the event loop.

        Using async functions can be useful for reducing race conditions.
        """
        self._impl.api._websock_interface.queue_message(
            _messages.SetSceneNodeClickableMessage(self._impl.name, True)
        )
        if self._impl.click_cb is None:
            self._impl.click_cb = []
        self._impl.click_cb.append(
            cast(
                Callable[
                    [SceneNodePointerEvent[_ClickableSceneNodeHandle]],
                    # `Union[X, Y]` instead of `X | Y` for Python 3.8 support.
                    Union[None, Coroutine],
                ],
                func,
            )
        )
        return func

    def remove_click_callback(
        self, callback: Literal["all"] | Callable = "all"
    ) -> None:
        """Remove click callbacks from scene node.

        Args:
            callback: Either "all" to remove all callbacks, or a specific callback function to remove.
        """
        if callback == "all":
            self._impl.click_cb.clear()
        else:
            self._impl.click_cb = [cb for cb in self._impl.click_cb if cb != callback]
        if len(self._impl.click_cb) == 0:
            self._impl.api._websock_interface.queue_message(
                _messages.SetSceneNodeClickableMessage(self._impl.name, False)
            )

    def remove(self) -> None:
        """Remove the node from the scene."""
        if len(self._impl.click_cb) > 0:
            # SetSceneNodeClickableMessage can still remain in the message
            # buffer, even if a scene node is removed. This could cause a new
            # scene node to be mistakenly considered clickable if it is made
            # with the same name. We ideally would fix this with better state
            # management... in the meantime we can just set the flag to False.
            self._impl.api._websock_interface.queue_message(
                _messages.SetSceneNodeClickableMessage(self._impl.name, False)
            )
        super().remove()


class CameraFrustumHandle(
    _ClickableSceneNodeHandle,
    _messages.CameraFrustumProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for camera frustums."""

    _image: np.ndarray | None
    _jpeg_quality: int | None

    @property
    def image(self) -> np.ndarray | None:
        """Current content of the image. Synchronized automatically when assigned."""
        return self._image

    @image.setter
    def image(self, image: np.ndarray | None) -> None:
        from ._scene_api import _encode_image_binary

        if image is None:
            self._image_data = None
            return

        if self.image_media_type is None:
            self.image_media_type = "image/png"

        self._image = image
        media_type, data = _encode_image_binary(
            image, self.image_media_type, jpeg_quality=self._jpeg_quality
        )
        self._image_data = data
        del media_type

    def compute_canonical_frustum_size(self) -> tuple[float, float, float]:
        """Compute the X, Y, and Z dimensions of the frustum if it had
        `.scale=1.0`. These dimensions will change whenever `.fov` or `.aspect`
        are changed.

        To set the distance between a frustum's origin and image plane to 1, we
        can run:

        .. code-block:: python

            frustum.scale = 1.0 / frustum.compute_canonical_frustum_size()[2]


        `.scale` is a unitless value that scales the X/Y/Z dimensions linearly.
        It aims to preserve the visual volume of the frustum regardless of the
        aspect ratio or FOV. This method allows more precise computation and
        control of the frustum's dimensions.
        """
        # Math used in the client implementation.
        y = np.tan(self.fov / 2.0)
        x = y * self.aspect
        z = 1.0
        volume_scale = np.cbrt((x * y * z) / 3.0)

        z /= volume_scale

        # x and y need to be doubled, since on the client they correspond to
        # NDC-style spans [-1, 1].
        return x * 2.0, y * 2.0, z


class DirectionalLightHandle(
    SceneNodeHandle,
    _messages.DirectionalLightProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for directional lights."""


class AmbientLightHandle(
    SceneNodeHandle,
    _messages.AmbientLightProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for ambient lights."""


class HemisphereLightHandle(
    SceneNodeHandle,
    _messages.HemisphereLightProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for hemisphere lights."""


class PointLightHandle(
    SceneNodeHandle,
    _messages.PointLightProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for point lights."""


class RectAreaLightHandle(
    SceneNodeHandle,
    _messages.RectAreaLightProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for rectangular area lights."""


class SpotLightHandle(
    SceneNodeHandle,
    _messages.SpotLightProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for spot lights."""


class PointCloudHandle(
    SceneNodeHandle,
    _messages.PointCloudProps,
    # We need `_HasCastArrayDtypes` here for type checking on `_cast_array_dtypes`.
    _OverridableScenePropApi if not TYPE_CHECKING else _HasCastArrayDtypes,
):
    """Handle for point clouds. Does not support click events."""

    @override
    def _cast_array_dtypes(
        self,
        prop_hints: Dict[str, Any],
        prop_name: str,
        value: np.ndarray,
    ) -> np.ndarray:
        """Casts assigned `points` based on the current value of `precision`."""
        if prop_name == "points":
            return value.astype(
                {"float16": np.float16, "float32": np.float32}[self.precision]
            )
        return super()._cast_array_dtypes(prop_hints, prop_name, value)


class BatchedAxesHandle(
    _ClickableSceneNodeHandle,
    _messages.BatchedAxesProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for batched coordinate frames."""


class FrameHandle(
    _ClickableSceneNodeHandle,
    _messages.FrameProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for coordinate frames."""


class MeshHandle(
    _ClickableSceneNodeHandle,
    _messages.MeshProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for mesh objects."""


class BatchedMeshHandle(
    _ClickableSceneNodeHandle,
    _messages.BatchedMeshesProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for batched mesh objects."""


class BatchedGlbHandle(
    _ClickableSceneNodeHandle,
    _messages.BatchedGlbProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for batched GLB objects."""


class GaussianSplatHandle(
    _ClickableSceneNodeHandle,
    _messages.GaussianSplatsProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for Gaussian splatting objects.

    **Work-in-progress.** Gaussian rendering is still under development.
    """


class MeshSkinnedHandle(
    _ClickableSceneNodeHandle,
    _messages.SkinnedMeshProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for skinned mesh objects."""

    def __init__(
        self, impl: _SceneNodeHandleState, bones: tuple[MeshSkinnedBoneHandle, ...]
    ):
        super().__init__(impl)
        self.bones = bones


@dataclasses.dataclass
class BoneState:
    name: str
    websock_interface: WebsockServer | WebsockClientConnection
    bone_index: int
    wxyz: np.ndarray
    position: np.ndarray


@dataclasses.dataclass
class MeshSkinnedBoneHandle:
    """Handle for reading and writing the poses of bones in a skinned mesh."""

    _impl: BoneState

    @property
    def wxyz(self) -> np.ndarray:
        """Orientation of the bone. This is the quaternion representation of the R
        in `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: tuple[float, float, float, float] | np.ndarray) -> None:
        from ._scene_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        wxyz_array = np.asarray(wxyz)
        if np.allclose(wxyz_cast, self._impl.wxyz):
            return
        self._impl.wxyz[:] = wxyz_array
        self._impl.websock_interface.queue_message(
            _messages.SetBoneOrientationMessage(
                self._impl.name, self._impl.bone_index, wxyz_cast
            )
        )

    @property
    def position(self) -> np.ndarray:
        """Position of the bone. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: tuple[float, float, float] | np.ndarray) -> None:
        from ._scene_api import cast_vector

        position_cast = cast_vector(position, 3)
        position_array = np.asarray(position)
        if np.allclose(position_array, self._impl.position):
            return
        self._impl.position[:] = position_array
        self._impl.websock_interface.queue_message(
            _messages.SetBonePositionMessage(
                self._impl.name, self._impl.bone_index, position_cast
            )
        )


class GridHandle(
    SceneNodeHandle,
    _messages.GridProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for grid objects."""


class LineSegmentsHandle(
    SceneNodeHandle,
    _messages.LineSegmentsProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for line segments objects."""


class SplineCatmullRomHandle(
    SceneNodeHandle,
    _messages.CatmullRomSplineProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for Catmull-Rom splines."""


class SplineCubicBezierHandle(
    SceneNodeHandle,
    _messages.CubicBezierSplineProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for cubic Bezier splines."""


class GlbHandle(
    _ClickableSceneNodeHandle,
    _messages.GlbProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for GLB objects."""


class ImageHandle(
    _ClickableSceneNodeHandle,
    _messages.ImageProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for 2D images, rendered in 3D."""

    _image: np.ndarray
    _jpeg_quality: int | None

    @property
    def image(self) -> np.ndarray:
        """Current content of the image. Synchronized automatically when assigned."""
        assert self._image is not None
        return self._image

    @image.setter
    def image(self, image: np.ndarray) -> None:
        from ._scene_api import _encode_image_binary

        self._image = image
        media_type, data = _encode_image_binary(
            image, self.media_type, jpeg_quality=self._jpeg_quality
        )
        self._data = data
        del media_type


class LabelHandle(
    SceneNodeHandle,
    _messages.LabelProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for 2D label objects. Does not support click events."""


@dataclasses.dataclass
class _TransformControlsState:
    last_updated: float
    update_cb: list[Callable[[TransformControlsHandle], None | Coroutine]]
    sync_cb: None | Callable[[ClientId, TransformControlsHandle], None] = None


class TransformControlsHandle(
    _ClickableSceneNodeHandle,
    _messages.TransformControlsProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Handle for interacting with transform control gizmos."""

    def __init__(self, impl: _SceneNodeHandleState, impl_aux: _TransformControlsState):
        super().__init__(impl)
        self._impl_aux = impl_aux

    @property
    def update_timestamp(self) -> float:
        return self._impl_aux.last_updated

    def on_update(
        self, func: Callable[[TransformControlsHandle], NoneOrCoroutine]
    ) -> Callable[[TransformControlsHandle], NoneOrCoroutine]:
        """Attach a callback for when the gizmo is moved.

        The callback can be either a standard function or an async function:
        - Standard functions (def) will be executed in a threadpool.
        - Async functions (async def) will be executed in the event loop.

        Using async functions can be useful for reducing race conditions.
        """
        self._impl_aux.update_cb.append(func)
        return func

    def remove_update_callback(
        self, callback: Literal["all"] | Callable = "all"
    ) -> None:
        """Remove update callbacks from the transform controls.

        Args:
            callback: Either "all" to remove all callbacks, or a specific callback function to remove.
        """
        if callback == "all":
            self._impl_aux.update_cb.clear()
        else:
            self._impl_aux.update_cb = [
                cb for cb in self._impl_aux.update_cb if cb != callback
            ]

    def remove(self) -> None:
        """Remove the node from the scene."""
        self._impl.api._handle_from_transform_controls_name.pop(self.name)
        super().remove()


class Gui3dContainerHandle(
    SceneNodeHandle,
    _messages.Gui3DProps,
    _OverridableScenePropApi if not TYPE_CHECKING else object,
):
    """Use as a context to place GUI elements into a 3D GUI container."""

    def __init__(self, impl: _SceneNodeHandleState, gui_api: GuiApi, container_id: str):
        super().__init__(impl)
        self._gui_api = gui_api
        self._container_id = container_id
        self._container_id_restore = None
        self._children = {}
        self._gui_api._container_handle_from_uuid[self._container_id] = self

    def __enter__(self) -> Gui3dContainerHandle:
        self._container_id_restore = self._gui_api._get_container_uuid()
        self._gui_api._set_container_uuid(self._container_id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_uuid(self._container_id_restore)
        self._container_id_restore = None

    def remove(self) -> None:
        """Permanently remove this GUI container from the visualizer."""

        # Call scene node remove.
        super().remove()

        # Clean up contained GUI elements.
        for child in tuple(self._children.values()):
            child.remove()
        self._gui_api._container_handle_from_uuid.pop(self._container_id)
