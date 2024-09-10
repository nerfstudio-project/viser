from __future__ import annotations

import copy
import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
)

import numpy as onp
import numpy.typing as onpt
from typing_extensions import get_type_hints

from . import _messages
from .infra._infra import WebsockClientConnection, WebsockServer

if TYPE_CHECKING:
    from ._gui_api import GuiApi
    from ._scene_api import SceneApi
    from ._viser import ClientHandle
    from .infra import ClientId


def colors_to_uint8(colors: onp.ndarray) -> onpt.NDArray[onp.uint8]:
    """Convert intensity values to uint8. We assume the range [0,1] for floats, and
    [0,255] for integers. Accepts any shape."""
    if colors.dtype != onp.uint8:
        if onp.issubdtype(colors.dtype, onp.floating):
            colors = onp.clip(colors * 255.0, 0, 255).astype(onp.uint8)
        if onp.issubdtype(colors.dtype, onp.integer):
            colors = onp.clip(colors, 0, 255).astype(onp.uint8)
    return colors


class _OverridablePropSettersAndGetters:
    def __setattr__(self, name: str, value: Any) -> None:
        handle = cast(SceneNodeHandle, self)
        # Get the value of the T TypeVar.
        if name in self._PropHints:
            # Help the user with some casting...
            hint = self._PropHints[name]
            if hint == onpt.NDArray[onp.float32]:
                value = value.astype(onp.float32)
            elif hint == onpt.NDArray[onp.uint8] and "color" in name:
                value = colors_to_uint8(value)

            setattr(handle._impl.props, name, value)
            handle._impl.api._websock_interface.queue_message(
                _messages.SceneNodeUpdateMessage(handle.name, {name: value})
            )
        else:
            return object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._PropHints:
            return getattr(self._impl.props, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


class _OverridablePropApi(
    _OverridablePropSettersAndGetters if not TYPE_CHECKING else object
):
    """Mixin that allows reading/assigning properties defined in each scene node message."""

    _PropHints: ClassVar[Dict[str, type]]

    def __init__(self) -> None:
        assert False

    def __init_subclass__(cls, PropClass: type):
        cls._PropHints = get_type_hints(PropClass)


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
    wxyz: onp.ndarray = dataclasses.field(
        default_factory=lambda: onp.array([1.0, 0.0, 0.0, 0.0])
    )
    position: onp.ndarray = dataclasses.field(
        default_factory=lambda: onp.array([0.0, 0.0, 0.0])
    )
    visible: bool = True
    # TODO: we should remove SceneNodeHandle as an argument here.
    click_cb: list[Callable[[SceneNodePointerEvent[SceneNodeHandle]], None]] | None = (
        None
    )


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
        wxyz: tuple[float, float, float, float] | onp.ndarray,
        position: tuple[float, float, float] | onp.ndarray,
        visible: bool,
    ) -> TSceneNodeHandle:
        """Create scene node: send state to client(s) and set up
        server-side state."""
        # Send message.
        assert isinstance(message, _messages.Message)
        api._websock_interface.queue_message(message)

        out = cls(_SceneNodeHandleState(name, copy.copy(message.props), api))
        api._handle_from_node_name[name] = out

        out.wxyz = wxyz
        out.position = position

        # Toggle visibility to make sure we send a
        # SetSceneNodeVisibilityMessage to the client.
        out._impl.visible = not visible
        out.visible = visible
        return out

    @property
    def wxyz(self) -> onp.ndarray:
        """Orientation of the scene node. This is the quaternion representation of the R
        in `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: tuple[float, float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        self._impl.wxyz = onp.asarray(wxyz)
        self._impl.api._websock_interface.queue_message(
            _messages.SetOrientationMessage(self._impl.name, wxyz_cast)
        )

    @property
    def position(self) -> onp.ndarray:
        """Position of the scene node. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: tuple[float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        position_cast = cast_vector(position, 3)
        self._impl.position = onp.asarray(position)
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


class _ClickableSceneNodeHandle(SceneNodeHandle):
    def on_click(
        self: TSceneNodeHandle,
        func: Callable[[SceneNodePointerEvent[TSceneNodeHandle]], None],
    ) -> Callable[[SceneNodePointerEvent[TSceneNodeHandle]], None]:
        """Attach a callback for when a scene node is clicked."""
        self._impl.api._websock_interface.queue_message(
            _messages.SetSceneNodeClickableMessage(self._impl.name, True)
        )
        if self._impl.click_cb is None:
            self._impl.click_cb = []
        self._impl.click_cb.append(func)  # type: ignore
        return func


class CameraFrustumHandle(
    _ClickableSceneNodeHandle,
    _messages.CameraFrustumProps,
    _OverridablePropApi,
    PropClass=_messages.CameraFrustumProps,
):
    """Handle for camera frustums."""


class DirectionalLightHandle(
    SceneNodeHandle,
    _messages.DirectionalLightProps,
    _OverridablePropApi,
    PropClass=_messages.DirectionalLightProps,
):
    """Handle for directional lights."""


class AmbientLightHandle(
    SceneNodeHandle,
    _messages.AmbientLightProps,
    _OverridablePropApi,
    PropClass=_messages.AmbientLightProps,
):
    """Handle for ambient lights."""


class HemisphereLightHandle(
    SceneNodeHandle,
    _messages.HemisphereLightProps,
    _OverridablePropApi,
    PropClass=_messages.HemisphereLightProps,
):
    """Handle for hemisphere lights."""


class PointLightHandle(
    SceneNodeHandle,
    _messages.PointLightProps,
    _OverridablePropApi,
    PropClass=_messages.PointLightProps,
):
    """Handle for point lights."""


class RectAreaLightHandle(
    SceneNodeHandle,
    _messages.RectAreaLightProps,
    _OverridablePropApi,
    PropClass=_messages.RectAreaLightProps,
):
    """Handle for rectangular area lights."""


class SpotLightHandle(
    SceneNodeHandle,
    _messages.SpotLightProps,
    _OverridablePropApi,
    PropClass=_messages.SpotLightProps,
):
    """Handle for spot lights."""


class PointCloudHandle(
    SceneNodeHandle,
    _messages.PointCloudProps,
    _OverridablePropApi,
    PropClass=_messages.PointCloudProps,
):
    """Handle for point clouds. Does not support click events."""


class BatchedAxesHandle(
    _ClickableSceneNodeHandle,
    _messages.BatchedAxesProps,
    _OverridablePropApi,
    PropClass=_messages.BatchedAxesProps,
):
    """Handle for batched coordinate frames."""


class FrameHandle(
    _ClickableSceneNodeHandle,
    _messages.FrameProps,
    _OverridablePropApi,
    PropClass=_messages.FrameProps,
):
    """Handle for coordinate frames."""


class MeshHandle(
    _ClickableSceneNodeHandle,
    _messages.MeshProps,
    _OverridablePropApi,
    PropClass=_messages.MeshProps,
):
    """Handle for mesh objects."""


class GaussianSplatHandle(
    _ClickableSceneNodeHandle,
    _messages.GaussianSplatsProps,
    _OverridablePropApi,
    PropClass=_messages.GaussianSplatsProps,
):
    """Handle for Gaussian splatting objects.

    **Work-in-progress.** Gaussian rendering is still under development.
    """


class MeshSkinnedHandle(
    _ClickableSceneNodeHandle,
    _messages.SkinnedMeshProps,
    _OverridablePropApi,
    PropClass=_messages.SkinnedMeshProps,
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
    wxyz: onp.ndarray
    position: onp.ndarray


@dataclasses.dataclass
class MeshSkinnedBoneHandle:
    """Handle for reading and writing the poses of bones in a skinned mesh."""

    _impl: BoneState

    @property
    def wxyz(self) -> onp.ndarray:
        """Orientation of the bone. This is the quaternion representation of the R
        in `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: tuple[float, float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        self._impl.wxyz = onp.asarray(wxyz)
        self._impl.websock_interface.queue_message(
            _messages.SetBoneOrientationMessage(
                self._impl.name, self._impl.bone_index, wxyz_cast
            )
        )

    @property
    def position(self) -> onp.ndarray:
        """Position of the bone. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: tuple[float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        position_cast = cast_vector(position, 3)
        self._impl.position = onp.asarray(position)
        self._impl.websock_interface.queue_message(
            _messages.SetBonePositionMessage(
                self._impl.name, self._impl.bone_index, position_cast
            )
        )


class GridHandle(
    SceneNodeHandle,
    _messages.GridProps,
    _OverridablePropApi,
    PropClass=_messages.GridProps,
):
    """Handle for grid objects."""


class SplineCatmullRomHandle(
    SceneNodeHandle,
    _messages.CatmullRomSplineProps,
    _OverridablePropApi,
    PropClass=_messages.CatmullRomSplineProps,
):
    """Handle for Catmull-Rom splines."""


class SplineCubicBezierHandle(
    SceneNodeHandle,
    _messages.CubicBezierSplineProps,
    _OverridablePropApi,
    PropClass=_messages.CubicBezierSplineProps,
):
    """Handle for cubic Bezier splines."""


class GlbHandle(
    _ClickableSceneNodeHandle,
    _messages.GlbProps,
    _OverridablePropApi,
    PropClass=_messages.GlbProps,
):
    """Handle for GLB objects."""


class ImageHandle(
    _ClickableSceneNodeHandle,
    _messages.ImageProps,
    _OverridablePropApi,
    PropClass=_messages.ImageProps,
):
    """Handle for 2D images, rendered in 3D."""


class LabelHandle(
    SceneNodeHandle,
    _messages.LabelProps,
    _OverridablePropApi,
    PropClass=_messages.LabelProps,
):
    """Handle for 2D label objects. Does not support click events."""


@dataclasses.dataclass
class _TransformControlsState:
    last_updated: float
    update_cb: list[Callable[[TransformControlsHandle], None]]
    sync_cb: None | Callable[[ClientId, TransformControlsHandle], None] = None


class TransformControlsHandle(
    _ClickableSceneNodeHandle,
    _messages.TransformControlsProps,
    _OverridablePropApi,
    PropClass=_messages.TransformControlsProps,
):
    """Handle for interacting with transform control gizmos."""

    def __init__(self, impl: _SceneNodeHandleState, impl_aux: _TransformControlsState):
        super().__init__(impl)
        self._impl_aux = impl_aux

    @property
    def update_timestamp(self) -> float:
        return self._impl_aux.last_updated

    def on_update(
        self, func: Callable[[TransformControlsHandle], None]
    ) -> Callable[[TransformControlsHandle], None]:
        """Attach a callback for when the gizmo is moved."""
        self._impl_aux.update_cb.append(func)
        return func


class Gui3dContainerHandle(
    SceneNodeHandle,
    _messages.Gui3DProps,
    _OverridablePropApi,
    PropClass=_messages.Gui3DProps,
):
    """Use as a context to place GUI elements into a 3D GUI container."""

    def __init__(self, impl: _SceneNodeHandleState, gui_api: GuiApi, container_id: str):
        super().__init__(impl)
        self._gui_api = gui_api
        self._container_id = container_id
        self._container_id_restore = None
        self._children = {}
        self._gui_api._container_handle_from_id[self._container_id] = self

    def __enter__(self) -> Gui3dContainerHandle:
        self._container_id_restore = self._gui_api._get_container_id()
        self._gui_api._set_container_id(self._container_id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def remove(self) -> None:
        """Permanently remove this GUI container from the visualizer."""

        # Call scene node remove.
        super().remove()

        # Clean up contained GUI elements.
        for child in tuple(self._children.values()):
            child.remove()
        self._gui_api._container_handle_from_id.pop(self._container_id)
