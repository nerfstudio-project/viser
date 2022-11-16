"""Message type definitions. This currently needs to be synchronized manully with the
client definitions in `WebsocketMessages.tsx`."""

import dataclasses
from typing import Any, ClassVar, Dict, List, Tuple

import msgpack
import numpy as onp


def _prepare_for_serialization(value: Any) -> Any:
    """Prepare any special types for serialization. Currently just maps numpy arrays to
    their underlying data buffers."""

    if isinstance(value, onp.ndarray):
        return value.data if value.data.c_contiguous else value.copy().data
    else:
        return value


class Message:
    """Base message type for controlling our viewer."""

    type: ClassVar[str]

    def serialize(self) -> bytes:
        mapping = {k: _prepare_for_serialization(v) for k, v in vars(self).items()}
        out = msgpack.packb({"type": self.type, **mapping})
        assert isinstance(out, bytes)
        return out


@dataclasses.dataclass
class CameraFrustumMessage(Message):
    type: ClassVar[str] = "camera_frustum"
    name: str
    fov: float
    aspect: float


@dataclasses.dataclass
class FrameMessage(Message):
    """Coordinate frame message.

    Position and orientation should follow a `T_parent_local` convention, which
    corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`."""

    type: ClassVar[str] = "frame"
    name: str
    xyzw: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    show_axes: bool = True


@dataclasses.dataclass
class PointCloudMessage(Message):
    type: ClassVar[str] = "point_cloud"
    name: str
    position_f32: onp.ndarray
    color_uint8: onp.ndarray
    point_size: float = 0.1

    def __post_init__(self):
        assert self.position_f32.dtype == onp.float32
        assert self.color_uint8.dtype == onp.uint8
        assert self.position_f32.shape == self.color_uint8.shape
        assert self.position_f32.shape[-1] == 3


@dataclasses.dataclass
class RemoveSceneNodeMessage(Message):
    """Remove a particular node from the scene."""

    type: ClassVar[str] = "remove_scene_node"
    name: str


@dataclasses.dataclass
class ResetSceneMessage(Message):
    """Reset scene."""

    type: ClassVar[str] = "reset_scene"
