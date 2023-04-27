from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as onp

from . import _messages

if TYPE_CHECKING:
    from ._message_api import ClientId, MessageApi


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


@dataclasses.dataclass
class SceneNodeHandle:
    """Handle for interacting with scene nodes."""

    _impl: _SceneNodeHandleState

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
        self._impl.api._queue(
            _messages.SetSceneNodeVisibilityMessage(self._impl.name, visible)
        )
        self._impl.visible = visible

    def remove(self) -> None:
        """Remove the node from the scene."""
        self._impl.api._queue(_messages.RemoveSceneNodeMessage(self._impl.name))


@dataclasses.dataclass
class _TransformControlsState:
    last_updated: float
    update_cb: List[Callable[[TransformControlsHandle], None]]
    sync_cb: Optional[Callable[[ClientId, TransformControlsHandle], None]] = None


@dataclasses.dataclass
class TransformControlsHandle(SceneNodeHandle):
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
