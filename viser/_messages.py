"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple, Union

import numpy as onp
import numpy.typing as onpt
from typing_extensions import Literal, override

from . import infra

class Message(infra.Message):
    @override
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
    look_at: Tuple[float, float, float]
    up_direction: Tuple[float, float, float]


@dataclasses.dataclass
class CameraFrustumMessage(Message):
    """Variant of CameraMessage used for visualizing camera frustums.

    OpenCV convention, +Z forward."""

    name: str
    fov: float
    aspect: float
    scale: float
    color: int
    image_media_type: Optional[Literal["image/jpeg", "image/png"]]
    image_base64_data: Optional[str]


@dataclasses.dataclass
class FrameMessage(Message):
    """Coordinate frame message.

    Position and orientation should follow a `T_parent_local` convention, which
    corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`."""

    name: str
    show_axes: bool = True
    axes_length: float = 0.5
    axes_radius: float = 0.025


@dataclasses.dataclass
class LabelMessage(Message):
    """Add a 2D label to the scene."""

    name: str
    text: str


@dataclasses.dataclass
class PointCloudMessage(Message):
    """Point cloud message.

    Positions are internally canonicalized to float32, colors to uint8.

    Float color inputs should be in the range [0,1], int color inputs should be in the
    range [0,255]."""

    name: str
    points: onpt.NDArray[onp.float32]
    colors: onpt.NDArray[onp.uint8]
    point_size: float = 0.1

    def __post_init__(self):
        # Check shapes.
        assert self.points.shape == self.colors.shape
        assert self.points.shape[-1] == 3

        # Check dtypes.
        assert self.points.dtype == onp.float32
        assert self.colors.dtype == onp.uint8


@dataclasses.dataclass
class MeshMessage(Message):
    """Mesh message.

    Vertices are internally canonicalized to float32, faces to uint32."""

    name: str
    vertices: onpt.NDArray[onp.float32]
    faces: onpt.NDArray[onp.uint32]
    color: int
    wireframe: bool
    side: Literal["front", "back", "double"] = "front"

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
class SetCameraPositionMessage(Message):
    """Server -> client message to set the camera's position."""

    position: Tuple[float, float, float]


@dataclasses.dataclass
class SetCameraUpDirectionMessage(Message):
    """Server -> client message to set the camera's up direction."""

    position: Tuple[float, float, float]


@dataclasses.dataclass
class SetCameraLookAtMessage(Message):
    """Server -> client message to set the camera's look-at point."""

    look_at: Tuple[float, float, float]


@dataclasses.dataclass
class SetCameraFovMessage(Message):
    """Server -> client message to set the camera's field of view."""

    fov: float


@dataclasses.dataclass
class SetOrientationMessage(Message):
    """Server -> client message to set a scene node's orientation.

    As with all other messages, transforms take the `T_parent_local` convention."""

    name: str
    wxyz: Tuple[float, float, float, float]


@dataclasses.dataclass
class SetPositionMessage(Message):
    """Server -> client message to set a scene node's position.

    As with all other messages, transforms take the `T_parent_local` convention."""

    name: str
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
class GuiSetVisibleMessage(Message):
    """Sent client->server when a GUI input is changed."""

    name: str
    visible: bool


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


@dataclasses.dataclass
class MessageGroupStart(Message):
    """Sent server->client to indicate the start of a message group."""


@dataclasses.dataclass
class MessageGroupEnd(Message):
    """Sent server->client to indicate the end of a message group."""


@dataclasses.dataclass
class ThemeConfigurationMessage(Message):
    """Message from server->client to configure parts of the GUI."""

    canvas_background_color: int
    show_titlebar: bool
    titlebar_content: str