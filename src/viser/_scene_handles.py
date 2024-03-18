# mypy: disable-error-code="assignment"
#
# Asymmetric properties are supported in Pyright, but not yet in mypy.
# - https://github.com/python/mypy/issues/3004
# - https://github.com/python/mypy/pull/11643
from __future__ import annotations

import dataclasses
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import numpy as onp

from . import _messages

if TYPE_CHECKING:
    from ._gui_api import GuiApi
    from ._gui_handles import SupportsRemoveProtocol
    from ._message_api import ClientId, MessageApi
    from ._viser import ClientHandle


@dataclasses.dataclass(frozen=True)
class ScenePointerEvent:
    """Event passed to pointer callbacks for the scene (currently only clicks)."""

    client: ClientHandle
    """Client that triggered this event."""
    client_id: int
    """ID of client that triggered this event."""
    event: Literal["click"]
    """Type of event that was triggered. Currently we only support clicks."""
    ray_origin: Tuple[float, float, float]
    """Origin of 3D ray corresponding to this click, in world coordinates."""
    ray_direction: Tuple[float, float, float]
    """Direction of 3D ray corresponding to this click, in world coordinates."""


TSceneNodeHandle = TypeVar("TSceneNodeHandle", bound="SceneNodeHandle")


@dataclasses.dataclass
class _SceneNodeHandleState:
    name: str
    api: MessageApi
    wxyz: onp.ndarray = dataclasses.field(
        default_factory=lambda: onp.array([1.0, 0.0, 0.0, 0.0])
    )
    position: onp.ndarray = dataclasses.field(
        default_factory=lambda: onp.array([0.0, 0.0, 0.0])
    )
    visible: bool = True
    # TODO: we should remove SceneNodeHandle as an argument here.
    click_cb: Optional[
        List[Callable[[SceneNodePointerEvent[SceneNodeHandle]], None]]
    ] = None


@dataclasses.dataclass
class SceneNodeHandle:
    """Handle base class for interacting with scene nodes."""

    _impl: _SceneNodeHandleState

    @classmethod
    def _make(
        cls: Type[TSceneNodeHandle],
        api: MessageApi,
        name: str,
        wxyz: Tuple[float, float, float, float] | onp.ndarray,
        position: Tuple[float, float, float] | onp.ndarray,
        visible: bool,
    ) -> TSceneNodeHandle:
        out = cls(_SceneNodeHandleState(name, api))
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
    def wxyz(self, wxyz: Tuple[float, float, float, float] | onp.ndarray) -> None:
        from ._message_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        self._impl.wxyz = onp.asarray(wxyz)
        self._impl.api._queue(
            _messages.SetOrientationMessage(self._impl.name, wxyz_cast)
        )

    @property
    def position(self) -> onp.ndarray:
        """Position of the scene node. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: Tuple[float, float, float] | onp.ndarray) -> None:
        from ._message_api import cast_vector

        position_cast = cast_vector(position, 3)
        self._impl.position = onp.asarray(position)
        self._impl.api._queue(
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
        self._impl.api._queue(
            _messages.SetSceneNodeVisibilityMessage(self._impl.name, visible)
        )
        self._impl.visible = visible

    def remove(self) -> None:
        """Remove the node from the scene."""
        self._impl.api._queue(_messages.RemoveSceneNodeMessage(self._impl.name))


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
    ray_origin: Tuple[float, float, float]
    """Origin of 3D ray corresponding to this click, in world coordinates."""
    ray_direction: Tuple[float, float, float]
    """Direction of 3D ray corresponding to this click, in world coordinates."""


@dataclasses.dataclass
class _ClickableSceneNodeHandle(SceneNodeHandle):
    def on_click(
        self: TSceneNodeHandle,
        func: Callable[[SceneNodePointerEvent[TSceneNodeHandle]], None],
    ) -> Callable[[SceneNodePointerEvent[TSceneNodeHandle]], None]:
        """Attach a callback for when a scene node is clicked."""
        self._impl.api._queue(
            _messages.SetSceneNodeClickableMessage(self._impl.name, True)
        )
        if self._impl.click_cb is None:
            self._impl.click_cb = []
        self._impl.click_cb.append(func)  # type: ignore
        return func


@dataclasses.dataclass
class CameraFrustumHandle(_ClickableSceneNodeHandle):
    """Handle for camera frustums."""


@dataclasses.dataclass
class PointCloudHandle(SceneNodeHandle):
    """Handle for point clouds. Does not support click events."""


@dataclasses.dataclass
class BatchedAxesHandle(_ClickableSceneNodeHandle):
    """Handle for batched coordinate frames."""


@dataclasses.dataclass
class FrameHandle(_ClickableSceneNodeHandle):
    """Handle for coordinate frames."""


@dataclasses.dataclass
class MeshHandle(_ClickableSceneNodeHandle):
    """Handle for mesh objects."""


@dataclasses.dataclass
class GaussianSplatHandle(_ClickableSceneNodeHandle):
    """Handle for Gaussian splatting objects."""


@dataclasses.dataclass
class GlbHandle(_ClickableSceneNodeHandle):
    """Handle for GLB objects."""


@dataclasses.dataclass
class ImageHandle(_ClickableSceneNodeHandle):
    """Handle for 2D images, rendered in 3D."""


@dataclasses.dataclass
class LabelHandle(SceneNodeHandle):
    """Handle for 2D label objects. Does not support click events."""


@dataclasses.dataclass
class _TransformControlsState:
    last_updated: float
    update_cb: List[Callable[[TransformControlsHandle], None]]
    sync_cb: Optional[Callable[[ClientId, TransformControlsHandle], None]] = None


@dataclasses.dataclass
class TransformControlsHandle(_ClickableSceneNodeHandle):
    """Handle for interacting with transform control gizmos."""

    _impl_aux: _TransformControlsState

    @property
    def update_timestamp(self) -> float:
        return self._impl_aux.last_updated

    def on_update(
        self, func: Callable[[TransformControlsHandle], None]
    ) -> Callable[[TransformControlsHandle], None]:
        """Attach a callback for when the gizmo is moved."""
        self._impl_aux.update_cb.append(func)
        return func


@dataclasses.dataclass
class Gui3dContainerHandle(SceneNodeHandle):
    """Use as a context to place GUI elements into a 3D GUI container."""

    _gui_api: GuiApi
    _container_id: str
    _container_id_restore: Optional[str] = None
    _children: Dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )

    def __enter__(self) -> Gui3dContainerHandle:
        self._container_id_restore = self._gui_api._get_container_id()
        self._gui_api._set_container_id(self._container_id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def __post_init__(self) -> None:
        self._gui_api._container_handle_from_id[self._container_id] = self

    def remove(self) -> None:
        """Permanently remove this GUI container from the visualizer."""

        # Call scene node remove.
        super().remove()

        # Clean up contained GUI elements.
        self._gui_api._container_handle_from_id.pop(self._container_id)
        for child in self._children.values():
            child.remove()
