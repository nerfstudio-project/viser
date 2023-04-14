from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar, cast

import numpy as onp

if TYPE_CHECKING:
    from ._message_api import ClientId, MessageApi

TVector = TypeVar("TVector", bound=tuple)


def _cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if isinstance(vector, tuple):
        assert len(vector) == length
        return cast(TVector, vector)
    else:
        assert cast(onp.ndarray, vector).shape == (length,)
        return cast(TVector, tuple(map(float, vector)))


@dataclasses.dataclass
class _SceneNodeHandleState:
    name: str
    api: MessageApi


@dataclasses.dataclass(frozen=True)
class SceneNodeHandle:
    """Handle for interacting with scene nodes."""

    _impl: _SceneNodeHandleState

    def set_transform(
        self,
        wxyz: Tuple[float, float, float, float] | onp.ndarray,
        position: Tuple[float, float, float] | onp.ndarray,
    ) -> SceneNodeHandle:
        """Set the 6D pose of the scene node."""
        self._impl.api.set_scene_node_transform(self._impl.name, wxyz, position)
        return self

    def set_visibility(self, visible: bool) -> SceneNodeHandle:
        """Set the visibility of the scene node."""
        self._impl.api.set_scene_node_visibility(self._impl.name, visible)
        return self

    def remove(self) -> None:
        """Remove the node from the scene."""
        self._impl.api.remove_scene_node(self._impl.name)


@dataclasses.dataclass
class _TransformControlsState:
    name: str
    api: MessageApi
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    last_updated: float

    update_cb: List[Callable[[TransformControlsHandle], None]]
    sync_cb: Optional[Callable[[ClientId, _TransformControlsState], None]] = None


@dataclasses.dataclass(frozen=True)
class TransformControlsState:
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    last_updated: float


@dataclasses.dataclass(frozen=True)
class TransformControlsHandle:
    """Handle for interacting with transform control gizmos."""

    _impl: _TransformControlsState

    def get_state(self) -> TransformControlsState:
        """Get the current state of the gizmo."""
        return TransformControlsState(
            self._impl.wxyz, self._impl.position, self._impl.last_updated
        )

    def on_update(
        self, func: Callable[[TransformControlsHandle], None]
    ) -> Callable[[TransformControlsHandle], None]:
        """Attach a callback for when the gizmo is moved."""
        self._impl.update_cb.append(func)
        return func

    def set_transform(
        self,
        wxyz: Tuple[float, float, float, float] | onp.ndarray,
        position: Tuple[float, float, float] | onp.ndarray,
    ) -> TransformControlsHandle:
        """Set the 6D pose of the gizmo."""
        self._impl.api.set_scene_node_transform(self._impl.name, wxyz, position)
        self._impl.wxyz = _cast_vector(wxyz, 4)
        self._impl.position = _cast_vector(position, 3)
        return self

    def remove(self) -> None:
        """Remove the node from the scene."""
        self._impl.api.remove_scene_node(self._impl.name)
