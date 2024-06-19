"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import dataclasses
import uuid
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as onp
import numpy.typing as onpt
from typing_extensions import Annotated, Literal, NotRequired, TypedDict, override

from . import infra, theme

GuiSliderMark = TypedDict("GuiSliderMark", {"value": float, "label": NotRequired[str]})


class Message(infra.Message):
    _tags: ClassVar[Tuple[str, ...]] = tuple()

    @override
    def redundancy_key(self) -> str:
        """Returns a unique key for this message, used for detecting redundant
        messages.

        For example: if we send 1000 GuiSetValue messages for the same GUI element, we
        should only keep the latest messages.
        """
        parts = [type(self).__name__]

        # Scene node manipulation messages all have a "name" field.
        node_name = getattr(self, "name", None)
        if node_name is not None:
            parts.append(node_name)

        # GUI messages all have an "id" field.
        node_name = getattr(self, "id", None)
        if node_name is not None:
            parts.append(node_name)

        return "_".join(parts)


T = TypeVar("T", bound=Type[Message])


def tag_class(tag: str) -> Callable[[T], T]:
    """Decorator for tagging a class with a `type` field."""

    def wrapper(cls: T) -> T:
        cls._tags = (cls._tags or ()) + (tag,)
        return cls

    return wrapper


@dataclasses.dataclass
class RunJavascriptMessage(Message):
    """Message for running some arbitrary Javascript on the client.
    We use this to set up the Plotly.js package, via the plotly.min.js source
    code."""

    source: str

    @override
    def redundancy_key(self) -> str:
        # Never cull these messages.
        return str(uuid.uuid4())


@dataclasses.dataclass
class NotificationMessage(Message):
    """Notification message."""

    id: str
    title: str
    body: str
    loading: bool
    with_close_button: bool
    auto_close: Union[int, Literal[False]]

@dataclasses.dataclass
class ClearNotificationMessage(Message):
    """Clear a specific notification."""
    id: str

    
@dataclasses.dataclass
class ClearAllNotificationMessage(Message):
    """Clear all open notification messages."""


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


# The list of scene pointer events supported by the viser frontend.
ScenePointerEventType = Literal["click", "rect-select"]


@dataclasses.dataclass
class ScenePointerMessage(Message):
    """Message for a raycast-like pointer in the scene.
    origin is the viewing camera position, in world coordinates.
    direction is the vector if a ray is projected from the camera through the clicked pixel,
    """

    # Later we can add `double_click`, `move`, `down`, `up`, etc.
    event_type: ScenePointerEventType
    ray_origin: Optional[Tuple[float, float, float]]
    ray_direction: Optional[Tuple[float, float, float]]
    screen_pos: List[Tuple[float, float]]


@dataclasses.dataclass
class ScenePointerEnableMessage(Message):
    """Message to enable/disable scene click events."""

    enable: bool
    event_type: ScenePointerEventType

    @override
    def redundancy_key(self) -> str:
        return (
            type(self).__name__ + "-" + self.event_type + "-" + str(self.enable).lower()
        )


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
class GlbMessage(Message):
    """GlTF message."""

    name: str
    glb_data: bytes
    scale: float


@dataclasses.dataclass
class FrameMessage(Message):
    """Coordinate frame message."""

    name: str
    show_axes: bool
    axes_length: float
    axes_radius: float
    origin_radius: float


@dataclasses.dataclass
class BatchedAxesMessage(Message):
    """Batched axes message.

    Positions and orientations should follow a `T_parent_local` convention, which
    corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`."""

    name: str
    wxyzs_batched: onpt.NDArray[onp.float32]
    positions_batched: onpt.NDArray[onp.float32]
    axes_length: float
    axes_radius: float


@dataclasses.dataclass
class GridMessage(Message):
    """Grid message. Helpful for visualizing things like ground planes."""

    name: str

    width: float
    height: float
    width_segments: int
    height_segments: int

    plane: Literal["xz", "xy", "yx", "yz", "zx", "zy"]

    cell_color: int
    cell_thickness: float
    cell_size: float

    section_color: int
    section_thickness: float
    section_size: float


@dataclasses.dataclass
class LabelMessage(Message):
    """Add a 2D label to the scene."""

    name: str
    text: str


@dataclasses.dataclass
class Gui3DMessage(Message):
    """Add a 3D gui element to the scene."""

    order: float
    name: str
    container_id: str


@dataclasses.dataclass
class PointCloudMessage(Message):
    """Point cloud message.

    Positions are internally canonicalized to float32, colors to uint8.

    Float color inputs should be in the range [0,1], int color inputs should be in the
    range [0,255]."""

    name: str
    points: onpt.NDArray[onp.float32]
    colors: onpt.NDArray[onp.uint8]
    point_size: float
    point_ball_norm: float

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

    color: Optional[int]
    vertex_colors: Optional[onpt.NDArray[onp.uint8]]

    wireframe: bool
    opacity: Optional[float]
    flat_shading: bool
    side: Literal["front", "back", "double"]
    material: Literal["standard", "toon3", "toon5"]

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
    base64_rgb: str
    base64_depth: Optional[str]


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
class SetSceneNodeClickableMessage(Message):
    """Set the clickability of a particular node in the scene."""

    name: str
    clickable: bool


@dataclasses.dataclass
class SceneNodeClickMessage(Message):
    """Message for clicked objects."""

    name: str
    ray_origin: Tuple[float, float, float]
    ray_direction: Tuple[float, float, float]


@dataclasses.dataclass
class ResetSceneMessage(Message):
    """Reset scene."""


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddFolderMessage(Message):
    order: float
    id: str
    label: str
    container_id: str
    expand_by_default: bool
    visible: bool


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddMarkdownMessage(Message):
    order: float
    id: str
    markdown: str
    container_id: str
    visible: bool


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddPlotlyMessage(Message):
    order: float
    id: str
    plotly_json_str: str
    aspect: float
    container_id: str
    visible: bool


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddTabGroupMessage(Message):
    order: float
    id: str
    container_id: str
    tab_labels: Tuple[str, ...]
    tab_icons_html: Tuple[Union[str, None], ...]
    tab_container_ids: Tuple[str, ...]
    visible: bool


@dataclasses.dataclass
class _GuiAddInputBase(Message):
    """Base message type containing fields commonly used by GUI inputs."""

    order: float
    id: str
    label: str
    container_id: str
    hint: Optional[str]
    value: Any
    visible: bool
    disabled: bool


@dataclasses.dataclass
class GuiModalMessage(Message):
    order: float
    id: str
    title: str


@dataclasses.dataclass
class GuiCloseModalMessage(Message):
    id: str


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddButtonMessage(_GuiAddInputBase):
    # All GUI elements currently need an `value` field.
    # This makes our job on the frontend easier.
    value: bool
    color: Optional[
        Literal[
            "dark",
            "gray",
            "red",
            "pink",
            "grape",
            "violet",
            "indigo",
            "blue",
            "cyan",
            "green",
            "lime",
            "yellow",
            "orange",
            "teal",
        ]
    ]
    icon_html: Optional[str]


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddUploadButtonMessage(_GuiAddInputBase):
    color: Optional[
        Literal[
            "dark",
            "gray",
            "red",
            "pink",
            "grape",
            "violet",
            "indigo",
            "blue",
            "cyan",
            "green",
            "lime",
            "yellow",
            "orange",
            "teal",
        ]
    ]
    icon_html: Optional[str]
    mime_type: str


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddSliderMessage(_GuiAddInputBase):
    min: float
    max: float
    step: Optional[float]
    value: float
    precision: int
    marks: Optional[Tuple[GuiSliderMark, ...]] = None


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddMultiSliderMessage(_GuiAddInputBase):
    min: float
    max: float
    step: Optional[float]
    min_range: Optional[float]
    precision: int
    fixed_endpoints: bool = False
    marks: Optional[Tuple[GuiSliderMark, ...]] = None


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddNumberMessage(_GuiAddInputBase):
    value: float
    precision: int
    step: float
    min: Optional[float]
    max: Optional[float]


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddRgbMessage(_GuiAddInputBase):
    value: Tuple[int, int, int]


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddRgbaMessage(_GuiAddInputBase):
    value: Tuple[int, int, int, int]


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddCheckboxMessage(_GuiAddInputBase):
    value: bool


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddVector2Message(_GuiAddInputBase):
    value: Tuple[float, float]
    min: Optional[Tuple[float, float]]
    max: Optional[Tuple[float, float]]
    step: float
    precision: int


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddVector3Message(_GuiAddInputBase):
    value: Tuple[float, float, float]
    min: Optional[Tuple[float, float, float]]
    max: Optional[Tuple[float, float, float]]
    step: float
    precision: int


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddTextMessage(_GuiAddInputBase):
    value: str


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddDropdownMessage(_GuiAddInputBase):
    value: str
    options: Tuple[str, ...]


@tag_class("GuiAddComponentMessage")
@dataclasses.dataclass
class GuiAddButtonGroupMessage(_GuiAddInputBase):
    value: str
    options: Tuple[str, ...]


@dataclasses.dataclass
class GuiRemoveMessage(Message):
    """Sent server->client to remove a GUI element."""

    id: str


@dataclasses.dataclass
class GuiUpdateMessage(Message):
    """Sent client<->server when any property of a GUI component is changed."""

    id: str
    updates: Annotated[
        Dict[str, Any],
        infra.TypeScriptAnnotationOverride("Partial<GuiAddComponentMessage>"),
    ]
    """Mapping from property name to new value."""

    @override
    def redundancy_key(self) -> str:
        return (
            type(self).__name__
            + "-"
            + self.id
            + "-"
            + ",".join(list(self.updates.keys()))
        )


@dataclasses.dataclass
class ThemeConfigurationMessage(Message):
    """Message from server->client to configure parts of the GUI."""

    titlebar_content: Optional[theme.TitlebarConfig]
    control_layout: Literal["floating", "collapsible", "fixed"]
    control_width: Literal["small", "medium", "large"]
    show_logo: bool
    show_share_button: bool
    dark_mode: bool
    colors: Optional[Tuple[str, str, str, str, str, str, str, str, str, str]]


@dataclasses.dataclass
class CatmullRomSplineMessage(Message):
    """Message from server->client carrying Catmull-Rom spline information."""

    name: str
    positions: Tuple[Tuple[float, float, float], ...]
    curve_type: Literal["centripetal", "chordal", "catmullrom"]
    tension: float
    closed: bool
    line_width: float
    color: int
    segments: Optional[int]


@dataclasses.dataclass
class CubicBezierSplineMessage(Message):
    """Message from server->client carrying Cubic Bezier spline information."""

    name: str
    positions: Tuple[Tuple[float, float, float], ...]
    control_points: Tuple[Tuple[float, float, float], ...]
    line_width: float
    color: int
    segments: Optional[int]


@dataclasses.dataclass
class GetRenderRequestMessage(Message):
    """Message from server->client requesting a render of the current viewport."""

    format: Literal["image/jpeg", "image/png"]
    height: int
    width: int
    quality: int


@dataclasses.dataclass
class GetRenderResponseMessage(Message):
    """Message from client->server carrying a render."""

    payload: bytes


@dataclasses.dataclass
class FileTransferStart(Message):
    """Signal that a file is about to be sent."""

    source_component_id: Optional[str]
    """Origin GUI component, used for client->server file uploads."""
    transfer_uuid: str
    filename: str
    mime_type: str
    part_count: int
    size_bytes: int

    @override
    def redundancy_key(self) -> str:
        return type(self).__name__ + "-" + self.transfer_uuid


@dataclasses.dataclass
class FileTransferPart(Message):
    """Send a file for clients to download or upload files from client."""

    # TODO: it would make sense to rename all "id" instances to "uuid" for GUI component ids.
    source_component_id: Optional[str]
    transfer_uuid: str
    part: int
    content: bytes

    @override
    def redundancy_key(self) -> str:
        return type(self).__name__ + "-" + self.transfer_uuid + "-" + str(self.part)


@dataclasses.dataclass
class FileTransferPartAck(Message):
    """Send a file for clients to download or upload files from client."""

    source_component_id: Optional[str]
    transfer_uuid: str
    transferred_bytes: int
    total_bytes: int

    @override
    def redundancy_key(self) -> str:
        return (
            type(self).__name__
            + "-"
            + self.transfer_uuid
            + "-"
            + str(self.transferred_bytes)
        )


@dataclasses.dataclass
class ShareUrlRequest(Message):
    """Message from client->server to connect to the share URL server."""


@dataclasses.dataclass
class ShareUrlUpdated(Message):
    """Message from server->client to indicate that the share URL has been updated."""

    share_url: Optional[str]


@dataclasses.dataclass
class ShareUrlDisconnect(Message):
    """Message from client->server to disconnect from the share URL server."""


@dataclasses.dataclass
class SetGuiPanelLabelMessage(Message):
    """Message from server->client to set the label of the GUI panel."""

    label: Optional[str]
