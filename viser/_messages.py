"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, cast

import msgpack
import numpy as onp
import numpy.typing as onpt
from typing_extensions import Literal

if TYPE_CHECKING:
    from ._server import ClientId
else:
    ClientId = Any


def _prepare_for_serialization(value: Any) -> Any:
    """Prepare any special types for serialization. Currently just maps numpy arrays to
    their underlying data buffers."""

    if isinstance(value, onp.ndarray):
        return value.data if value.data.c_contiguous else value.copy().data
    else:
        return value


class Message:
    """Base message type for controlling our viewer."""

    excluded_self_client: Optional[ClientId] = None
    """Don't send this message to a particular client. Example of when this is useful:
    for synchronizing GUI stuff, we want to """

    def serialize(self) -> bytes:
        """Convert a Python Message object into bytes."""
        mapping = {k: _prepare_for_serialization(v) for k, v in vars(self).items()}
        out = msgpack.packb({"type": type(self).__name__, **mapping})
        assert isinstance(out, bytes)
        return out

    @staticmethod
    def deserialize(message: bytes) -> Message:
        """Convert bytes into a Python Message object."""
        mapping = msgpack.unpackb(message)

        # msgpack deserializes to lists by default, but all of our annotations use
        # tuples.
        mapping = {
            k: tuple(v) if isinstance(v, list) else v for k, v in mapping.items()
        }
        message_type = Message._subclass_from_type_string()[
            cast(str, mapping.pop("type"))
        ]
        return message_type(**mapping)

    @staticmethod
    @functools.lru_cache
    def _subclass_from_type_string() -> Dict[str, Type[Message]]:
        subclasses = Message.get_subclasses()
        return {s.__name__: s for s in subclasses}

    @staticmethod
    def get_subclasses() -> List[Type[Message]]:
        """Recursively get message subclasses."""

        def _get_subclasses(typ: Type[Message]) -> List[Type[Message]]:
            out = []
            for sub in typ.__subclasses__():
                out.append(sub)
                out.extend(_get_subclasses(sub))
            return out

        return _get_subclasses(Message)

    def redundancy_key(self) -> str:
        """Returns a unique key for this message, used for detecting redundant
        messages.

        For example: if we send 1000 GuiSetValue messages for the same gui element, we
        should only keep the latest messages.
        """
        parts = [type(self).__name__]

        # GUI and scene node manipulation messages all have a "name" field.
        node_name = getattr(self, "name", None)
        if node_name is not None:
            parts.append(node_name)

        return "_".join(parts)


@dataclasses.dataclass
class ViewerCameraMessage(Message):
    """Message for a posed viewer camera.
    Pose is in the form T_world_camera, OpenCV convention, +Z forward."""

    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    fov: float
    aspect: float


@dataclasses.dataclass
class CameraFrustumMessage(Message):
    """Variant of CameraMessage used for visualizing camera frustums.

    OpenCV convention, +Z forward."""

    name: str
    fov: float
    aspect: float
    scale: float
    color: int


@dataclasses.dataclass
class FrameMessage(Message):
    """Coordinate frame message.

    Position and orientation should follow a `T_parent_local` convention, which
    corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`."""

    name: str
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    show_axes: bool = True
    axes_length: float = 0.5
    axes_radius: float = 0.025


@dataclasses.dataclass
class PointCloudMessage(Message):
    """Point cloud message.

    Positions are internally canonicalized to float32, colors to uint8.

    Float color inputs should be in the range [0,1], int color inputs should be in the
    range [0,255]."""

    name: str
    position: onpt.NDArray[onp.float32]
    color: onpt.NDArray[onp.uint8]
    point_size: float = 0.1

    def __post_init__(self):
        # Check shapes.
        assert self.position.shape == self.color.shape
        assert self.position.shape[-1] == 3

        # Check dtypes.
        assert self.position.dtype == onp.float32
        assert self.color.dtype == onp.uint8


@dataclasses.dataclass
class MeshMessage(Message):
    """Mesh message.

    Vertices are internally canonicalized to float32, faces to uint32."""

    name: str
    vertices: onpt.NDArray[onp.float32]
    faces: onpt.NDArray[onp.uint32]
    color: int
    wireframe: bool

    def __post_init__(self):
        # Check shapes.
        assert self.vertices.shape[-1] == 3
        assert self.faces.shape[-1] == 3


@dataclasses.dataclass
class TransformControlsMessage(Message):
    """Message for transform gizmos."""

    name: str
    scale: float
    line_width: float
    fixed: bool
    auto_transform: bool
    active_axes: Tuple[bool, bool, bool]
    disable_axes: bool
    disable_sliders: bool
    disable_rotations: bool
    translation_limits: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]
    rotation_limits: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]
    depth_test: bool
    opacity: float


@dataclasses.dataclass
class SetTransformMessage(Message):
    """Server -> client message to set a scene node's pose.

    As with all other messages, transforms take the `T_parent_local` convention."""

    name: str
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]


@dataclasses.dataclass
class TransformControlsUpdateMessage(Message):
    """Client -> server message when a transform control is updated.

    As with all other messages, transforms take the `T_parent_local` convention."""

    name: str
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]


@dataclasses.dataclass
class BackgroundImageMessage(Message):
    """Message for rendering a background image."""

    media_type: Literal["image/jpeg", "image/png"]
    base64_data: str


@dataclasses.dataclass
class ImageMessage(Message):
    """Message for rendering 2D images."""

    name: str
    media_type: Literal["image/jpeg", "image/png"]
    base64_data: str
    render_width: float
    render_height: float


@dataclasses.dataclass
class RemoveSceneNodeMessage(Message):
    """Remove a particular node from the scene."""

    name: str


@dataclasses.dataclass
class SetSceneNodeVisibilityMessage(Message):
    """Set the visibility of a particular node in the scene."""

    name: str
    visible: bool


@dataclasses.dataclass
class ResetSceneMessage(Message):
    """Reset scene."""


@dataclasses.dataclass
class GuiAddMessage(Message):
    """Sent server->client to add a new GUI input."""

    name: str
    folder_labels: Tuple[str, ...]
    leva_conf: Any


@dataclasses.dataclass
class GuiRemoveMessage(Message):
    """Sent server->client to add a new GUI input."""

    name: str


@dataclasses.dataclass
class GuiUpdateMessage(Message):
    """Sent client->server when a GUI input is changed."""

    name: str
    value: Any


@dataclasses.dataclass
class GuiSetHiddenMessage(Message):
    """Sent client->server when a GUI input is changed."""

    name: str
    hidden: bool


@dataclasses.dataclass
class GuiSetValueMessage(Message):
    """Sent server->client to set the value of a particular input."""

    name: str
    value: Any


@dataclasses.dataclass
class GuiSetLevaConfMessage(Message):
    """Sent server->client to override some part of an input's Leva config."""

    name: str
    leva_conf: Any
