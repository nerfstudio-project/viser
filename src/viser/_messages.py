"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal, override

from . import infra, theme


@dataclasses.dataclass(frozen=True)
class GuiSliderMark:
    value: float
    label: Optional[str]


Color = Literal[
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


TagLiteral = Literal["GuiComponentMessage", "SceneNodeMessage"]


class Message(infra.Message):
    _tags: ClassVar[Tuple[TagLiteral, ...]] = tuple()

    @override
    def redundancy_key(self) -> str:
        """Returns a unique key for this message, used for detecting redundant
        messages.

        For example: if we send 1000 GUI value updates for the same GUI
        element, we should only keep the latest message.
        """
        parts = [type(self).__name__]

        # Scene node manipulation messages all have a "name" field.
        node_name = getattr(self, "name", None)
        if node_name is not None:
            parts.append(node_name)

        # GUI and notification messages all have an "uuid" field.
        node_name = getattr(self, "uuid", None)
        if node_name is not None:
            parts.append(node_name)

        return "_".join(parts)

    @classmethod
    def __init_subclass__(cls, tag: TagLiteral | None = None):
        """Tag will be used to create a union type in TypeScript."""
        super().__init_subclass__()
        if tag is not None:
            cls._tags = cls._tags + (tag,)


T = TypeVar("T", bound=Type[Message])


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

    mode: Literal["show", "update"]
    uuid: str
    props: NotificationProps


@dataclasses.dataclass
class NotificationProps:
    title: str
    """Title of the notification. Synchronized automatically when assigned."""
    body: str
    """Body text of the notification. Synchronized automatically when assigned."""
    loading: bool
    """Whether to show a loading indicator. Synchronized automatically when assigned."""
    with_close_button: bool
    """Whether to show a close button. Synchronized automatically when assigned."""
    auto_close: Union[int, Literal[False]]
    """Time in milliseconds after which the notification should auto-close, or False to disable auto-close. Synchronized automatically when assigned."""
    color: Optional[Color]
    """Color of the notification. Synchronized automatically when assigned."""


@dataclasses.dataclass
class RemoveNotificationMessage(Message):
    """Remove a specific notification."""

    uuid: str


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
    screen_pos: Tuple[Tuple[float, float], ...]


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
class CameraFrustumMessage(Message, tag="SceneNodeMessage"):
    """Variant of CameraMessage used for visualizing camera frustums.

    OpenCV convention, +Z forward."""

    name: str
    props: CameraFrustumProps


@dataclasses.dataclass
class CameraFrustumProps:
    fov: float
    """Field of view of the camera (in radians). Synchronized automatically when assigned."""
    aspect: float
    """Aspect ratio of the camera (width over height). Synchronized automatically when assigned."""
    scale: float
    """Scale factor for the size of the frustum. Synchronized automatically when assigned."""
    color: Tuple[int, int, int]
    """Color of the frustum as RGB integers. Synchronized automatically when assigned."""
    image_media_type: Optional[Literal["image/jpeg", "image/png"]]
    """Format of the provided image ('image/jpeg' or 'image/png'). Synchronized automatically when assigned."""
    image_binary: Optional[bytes]
    """Optional image to be displayed on the frustum. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GlbMessage(Message, tag="SceneNodeMessage"):
    """GlTF message."""

    name: str
    props: GlbProps


@dataclasses.dataclass
class GlbProps:
    glb_data: bytes
    """A binary payload containing the GLB data. Synchronized automatically when assigned."""
    scale: float
    """A scale for resizing the GLB asset. Synchronized automatically when assigned."""


@dataclasses.dataclass
class FrameMessage(Message, tag="SceneNodeMessage"):
    """Coordinate frame message."""

    name: str
    props: FrameProps


@dataclasses.dataclass
class FrameProps:
    show_axes: bool
    """Boolean to indicate whether to show the frame as a set of axes + origin sphere. Synchronized automatically when assigned."""
    axes_length: float
    """Length of each axis. Synchronized automatically when assigned."""
    axes_radius: float
    """Radius of each axis. Synchronized automatically when assigned."""
    origin_radius: float
    """Radius of the origin sphere. Synchronized automatically when assigned."""


@dataclasses.dataclass
class BatchedAxesMessage(Message, tag="SceneNodeMessage"):
    """Batched axes message.

    Positions and orientations should follow a `T_parent_local` convention, which
    corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`."""

    name: str
    props: BatchedAxesProps


@dataclasses.dataclass
class BatchedAxesProps:
    wxyzs_batched: npt.NDArray[np.float32]
    """Float array of shape (N,4) representing quaternion rotations. Synchronized automatically when assigned."""
    positions_batched: npt.NDArray[np.float32]
    """Float array of shape (N,3) representing positions. Synchronized automatically when assigned."""
    axes_length: float
    """Length of each axis. Synchronized automatically when assigned."""
    axes_radius: float
    """Radius of each axis. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GridMessage(Message, tag="SceneNodeMessage"):
    """Grid message. Helpful for visualizing things like ground planes."""

    name: str
    props: GridProps


@dataclasses.dataclass
class GridProps:
    width: float
    """Width of the grid. Synchronized automatically when assigned."""
    height: float
    """Height of the grid. Synchronized automatically when assigned."""
    width_segments: int
    """Number of segments along the width. Synchronized automatically when assigned."""
    height_segments: int
    """Number of segments along the height. Synchronized automatically when assigned."""
    plane: Literal["xz", "xy", "yx", "yz", "zx", "zy"]
    """The plane in which the grid is oriented. Synchronized automatically when assigned."""
    cell_color: Tuple[int, int, int]
    """Color of the grid cells as RGB integers. Synchronized automatically when assigned."""
    cell_thickness: float
    """Thickness of the grid lines. Synchronized automatically when assigned."""
    cell_size: float
    """Size of each cell in the grid. Synchronized automatically when assigned."""
    section_color: Tuple[int, int, int]
    """Color of the grid sections as RGB integers. Synchronized automatically when assigned."""
    section_thickness: float
    """Thickness of the section lines. Synchronized automatically when assigned."""
    section_size: float
    """Size of each section in the grid. Synchronized automatically when assigned."""


@dataclasses.dataclass
class LabelMessage(Message, tag="SceneNodeMessage"):
    """Add a 2D label to the scene."""

    name: str
    props: LabelProps


@dataclasses.dataclass
class LabelProps:
    text: str
    """Text content of the label. Synchronized automatically when assigned."""


@dataclasses.dataclass
class Gui3DMessage(Message, tag="SceneNodeMessage"):
    """Add a 3D gui element to the scene."""

    name: str
    props: Gui3DProps


@dataclasses.dataclass
class Gui3DProps:
    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    container_uuid: str
    """Identifier for the container. Synchronized automatically when assigned."""


@dataclasses.dataclass
class PointCloudMessage(Message, tag="SceneNodeMessage"):
    """Point cloud message.

    Positions are internally canonicalized to float32, colors to uint8.

    Float color inputs should be in the range [0,1], int color inputs should be in the
    range [0,255]."""

    name: str
    props: PointCloudProps


@dataclasses.dataclass
class PointCloudProps:
    points: npt.NDArray[np.float16]
    """Location of points. Should have shape (N, 3). Synchronized automatically when assigned."""
    colors: npt.NDArray[np.uint8]
    """Colors of points. Should have shape (N, 3) or (3,). Synchronized automatically when assigned."""
    point_size: float
    """Size of each point. Synchronized automatically when assigned."""
    point_ball_norm: float
    """Norm value determining the shape of each point. Synchronized automatically when assigned."""

    def __post_init__(self):
        # Check shapes.
        assert self.points.shape == self.colors.shape
        assert self.points.shape[-1] == 3

        # Check dtypes.
        assert self.points.dtype == np.float16
        assert self.colors.dtype == np.uint8


@dataclasses.dataclass
@dataclasses.dataclass
class DirectionalLightMessage(Message, tag="SceneNodeMessage"):
    """Directional light message."""

    name: str
    props: DirectionalLightProps


@dataclasses.dataclass
class DirectionalLightProps:
    color: Tuple[int, int, int]
    """Color of the directional light. Synchronized automatically when assigned."""
    intensity: float
    """Intensity of the directional light. Synchronized automatically when assigned."""


@dataclasses.dataclass
class AmbientLightMessage(Message, tag="SceneNodeMessage"):
    """Ambient light message."""

    name: str
    props: AmbientLightProps


@dataclasses.dataclass
class AmbientLightProps:
    color: Tuple[int, int, int]
    """Color of the ambient light. Synchronized automatically when assigned."""
    intensity: float
    """Intensity of the ambient light. Synchronized automatically when assigned."""


@dataclasses.dataclass
class HemisphereLightMessage(Message, tag="SceneNodeMessage"):
    """Hemisphere light message."""

    name: str
    props: HemisphereLightProps


@dataclasses.dataclass
class HemisphereLightProps:
    sky_color: Tuple[int, int, int]
    """Sky color of the hemisphere light. Synchronized automatically when assigned."""
    ground_color: Tuple[int, int, int]
    """Ground color of the hemisphere light. Synchronized automatically when assigned."""
    intensity: float
    """Intensity of the hemisphere light. Synchronized automatically when assigned."""


@dataclasses.dataclass
class PointLightMessage(Message, tag="SceneNodeMessage"):
    """Point light message."""

    name: str
    props: PointLightProps


@dataclasses.dataclass
class PointLightProps:
    color: Tuple[int, int, int]
    """Color of the point light. Synchronized automatically when assigned."""
    intensity: float
    """Intensity of the point light. Synchronized automatically when assigned."""
    distance: float
    """Distance of the point light. Synchronized automatically when assigned."""
    decay: float
    """Decay of the point light. Synchronized automatically when assigned."""


@dataclasses.dataclass
class RectAreaLightMessage(Message, tag="SceneNodeMessage"):
    """Rectangular Area light message."""

    name: str
    props: RectAreaLightProps


@dataclasses.dataclass
class RectAreaLightProps:
    color: Tuple[int, int, int]
    """Color of the rectangular area light. Synchronized automatically when assigned."""
    intensity: float
    """Intensity of the rectangular area light. Synchronized automatically when assigned."""
    width: float
    """Width of the rectangular area light. Synchronized automatically when assigned."""
    height: float
    """Height of the rectangular area light. Synchronized automatically when assigned."""


@dataclasses.dataclass
class SpotLightMessage(Message, tag="SceneNodeMessage"):
    """Spot light message."""

    name: str
    props: SpotLightProps


@dataclasses.dataclass
class SpotLightProps:
    color: Tuple[int, int, int]
    """Color of the spot light. Synchronized automatically when assigned."""
    intensity: float
    """Intensity of the spot light. Synchronized automatically when assigned."""
    distance: float
    """Distance of the spot light. Synchronized automatically when assigned."""
    angle: float
    """Angle of the spot light. Synchronized automatically when assigned."""
    penumbra: float
    """Penumbra of the spot light. Synchronized automatically when assigned."""
    decay: float
    """Decay of the spot light. Synchronized automatically when assigned."""

    def __post_init__(self):
        assert self.angle <= np.pi / 2
        assert self.angle >= 0


@dataclasses.dataclass
class EnvironmentMapMessage(Message):
    """Environment Map message."""

    hdri: Optional[
        Literal[
            "apartment",
            "city",
            "dawn",
            "forest",
            "lobby",
            "night",
            "park",
            "studio",
            "sunset",
            "warehouse",
        ]
    ]
    background: bool
    background_blurriness: float
    background_intensity: float
    background_rotation: tuple[float, float, float]
    environment_intensity: float
    environment_rotation: tuple[float, float, float]


@dataclasses.dataclass
class EnableLightsMessage(Message):
    """Spot light message."""

    enabled: bool


@dataclasses.dataclass
class MeshMessage(Message, tag="SceneNodeMessage"):
    """Mesh message.

    Vertices are internally canonicalized to float32, faces to uint32."""

    name: str
    props: MeshProps


@dataclasses.dataclass
class MeshProps:
    vertices: npt.NDArray[np.float32]
    """A numpy array of vertex positions. Should have shape (V, 3). Synchronized automatically when assigned."""
    faces: npt.NDArray[np.uint32]
    """A numpy array of faces, where each face is represented by indices of vertices. Should have shape (F, 3). Synchronized automatically when assigned."""
    color: Optional[Tuple[int, int, int]]
    """Color of the mesh as RGB integers. Synchronized automatically when assigned."""
    wireframe: bool
    """Boolean indicating if the mesh should be rendered as a wireframe. Synchronized automatically when assigned."""
    opacity: Optional[float]
    """Opacity of the mesh. None means opaque. Synchronized automatically when assigned."""
    flat_shading: bool
    """Whether to do flat shading. Synchronized automatically when assigned."""
    side: Literal["front", "back", "double"]
    """Side of the surface to render. Synchronized automatically when assigned."""
    material: Literal["standard", "toon3", "toon5"]
    """Material type of the mesh. Synchronized automatically when assigned."""

    def __post_init__(self):
        # Check shapes.
        assert self.vertices.shape[-1] == 3
        assert self.faces.shape[-1] == 3


@dataclasses.dataclass
class SkinnedMeshMessage(Message, tag="SceneNodeMessage"):
    """Skinned mesh message."""

    name: str
    props: SkinnedMeshProps


@dataclasses.dataclass
class SkinnedMeshProps(MeshProps):
    """Mesh message.

    Vertices are internally canonicalized to float32, faces to uint32."""

    bone_wxyzs: npt.NDArray[np.float32]
    """Array of quaternions representing bone orientations (B, 4). Synchronized automatically when assigned."""
    bone_positions: npt.NDArray[np.float32]
    """Array of positions representing bone positions (B, 3). Synchronized automatically when assigned."""
    skin_indices: npt.NDArray[np.uint16]
    """Array of skin indices. Should have shape (V, 4). Synchronized automatically when assigned."""
    skin_weights: npt.NDArray[np.float32]
    """Array of skin weights. Should have shape (V, 4). Synchronized automatically when assigned."""

    def __post_init__(self):
        # Check shapes.
        assert self.bone_wxyzs.shape[-1] == 4
        assert self.bone_positions.shape[-1] == 3
        assert self.bone_wxyzs.shape[0] == self.bone_positions.shape[0]
        assert self.vertices.shape[-1] == 3
        assert self.faces.shape[-1] == 3
        assert self.skin_weights is not None
        assert (
            self.skin_indices.shape
            == self.skin_weights.shape
            == (self.vertices.shape[0], 4)
        )


@dataclasses.dataclass
class SetBoneOrientationMessage(Message):
    """Server -> client message to set a skinned mesh bone's orientation.

    As with all other messages, transforms take the `T_parent_local` convention."""

    name: str
    bone_index: int
    wxyz: Tuple[float, float, float, float]

    @override
    def redundancy_key(self) -> str:
        return type(self).__name__ + "-" + self.name + "-" + str(self.bone_index)


@dataclasses.dataclass
class SetBonePositionMessage(Message):
    """Server -> client message to set a skinned mesh bone's position.

    As with all other messages, transforms take the `T_parent_local` convention."""

    name: str
    bone_index: int
    position: Tuple[float, float, float]

    @override
    def redundancy_key(self) -> str:
        return type(self).__name__ + "-" + self.name + "-" + str(self.bone_index)


@dataclasses.dataclass
class TransformControlsMessage(Message, tag="SceneNodeMessage"):
    """Message for transform gizmos."""

    name: str
    props: TransformControlsProps


@dataclasses.dataclass
class TransformControlsProps:
    scale: float
    """Scale of the transform controls. Synchronized automatically when assigned."""
    line_width: float
    """Width of the lines used in the gizmo. Synchronized automatically when assigned."""
    fixed: bool
    """Boolean indicating if the gizmo should be fixed in position. Synchronized automatically when assigned."""
    auto_transform: bool
    """Whether the transform should be applied automatically. Synchronized automatically when assigned."""
    active_axes: Tuple[bool, bool, bool]
    """Tuple of booleans indicating active axes. Synchronized automatically when assigned."""
    disable_axes: bool
    """Boolean to disable axes interaction. Synchronized automatically when assigned."""
    disable_sliders: bool
    """Boolean to disable slider interaction. Synchronized automatically when assigned."""
    disable_rotations: bool
    """Boolean to disable rotation interaction. Synchronized automatically when assigned."""
    translation_limits: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]
    """Limits for translation. Synchronized automatically when assigned."""
    rotation_limits: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]
    """Limits for rotation. Synchronized automatically when assigned."""
    depth_test: bool
    """Boolean indicating if depth testing should be used when rendering. Synchronized automatically when assigned."""
    opacity: float
    """Opacity of the gizmo. Synchronized automatically when assigned."""


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
    rgb_bytes: bytes
    depth_bytes: Optional[bytes]


@dataclasses.dataclass
class ImageMessage(Message, tag="SceneNodeMessage"):
    """Message for rendering 2D images."""

    name: str
    props: ImageProps


@dataclasses.dataclass
class ImageProps:
    media_type: Literal["image/jpeg", "image/png"]
    """Format of the provided image ('image/jpeg' or 'image/png'). Synchronized automatically when assigned."""
    data: bytes
    """Binary data of the image. Synchronized automatically when assigned."""
    render_width: float
    """Width at which the image should be rendered in the scene. Synchronized automatically when assigned."""
    render_height: float
    """Height at which the image should be rendered in the scene. Synchronized automatically when assigned."""


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
    instance_index: Optional[int]
    """Instance index. Currently only used for batched axes."""
    ray_origin: Tuple[float, float, float]
    ray_direction: Tuple[float, float, float]
    screen_pos: Tuple[float, float]


@dataclasses.dataclass
class ResetSceneMessage(Message):
    """Reset scene."""


@dataclasses.dataclass
class ResetGuiMessage(Message):
    """Reset GUI."""


@dataclasses.dataclass
class GuiBaseProps:
    """Base message type containing fields commonly used by GUI inputs."""

    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    label: str
    """Label text for the GUI element. Synchronized automatically when assigned."""
    hint: Optional[str]
    """Optional hint text for the GUI element. Synchronized automatically when assigned."""
    visible: bool
    """Visibility state of the GUI element. Synchronized automatically when assigned."""
    disabled: bool
    """Disabled state of the GUI element. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiFolderProps:
    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    label: str
    """Label text for the GUI folder. Synchronized automatically when assigned."""
    visible: bool
    """Visibility state of the GUI folder. Synchronized automatically when assigned."""
    expand_by_default: bool
    """Whether the folder should be expanded by default. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiFolderMessage(Message, tag="GuiComponentMessage"):
    uuid: str
    container_uuid: str
    props: GuiFolderProps


@dataclasses.dataclass
class GuiMarkdownProps:
    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    _markdown: str
    """(Private) Markdown content to be displayed. Synchronized automatically when assigned."""
    visible: bool
    """Visibility state of the markdown element. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiMarkdownMessage(Message, tag="GuiComponentMessage"):
    uuid: str
    container_uuid: str
    props: GuiMarkdownProps


@dataclasses.dataclass
class GuiProgressBarProps:
    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    animated: bool
    """Whether the progress bar should be animated. Synchronized automatically when assigned."""
    color: Optional[Color]
    """Color of the progress bar. Synchronized automatically when assigned."""
    visible: bool
    """Visibility state of the progress bar. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiProgressBarMessage(Message, tag="GuiComponentMessage"):
    value: float
    uuid: str
    container_uuid: str
    props: GuiProgressBarProps


@dataclasses.dataclass
class GuiPlotlyProps:
    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    _plotly_json_str: str
    """(Private) JSON string representation of the Plotly figure. Synchronized automatically when assigned."""
    aspect: float
    """Aspect ratio of the plot. Synchronized automatically when assigned."""
    visible: bool
    """Visibility state of the plot. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiPlotlyMessage(Message, tag="GuiComponentMessage"):
    uuid: str
    container_uuid: str
    props: GuiPlotlyProps


@dataclasses.dataclass
class GuiTabGroupProps:
    _tab_labels: Tuple[str, ...]
    """(Private) Tuple of labels for each tab. Synchronized automatically when assigned."""
    _tab_icons_html: Tuple[Union[str, None], ...]
    """(Private) Tuple of HTML strings for icons of each tab, or None if no icon. Synchronized automatically when assigned."""
    _tab_container_ids: Tuple[str, ...]
    """(Private) Tuple of container IDs for each tab. Synchronized automatically when assigned."""
    order: float
    """Order value for arranging GUI elements. Synchronized automatically when assigned."""
    visible: bool
    """Visibility state of the tab group. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiTabGroupMessage(Message, tag="GuiComponentMessage"):
    uuid: str
    container_uuid: str
    props: GuiTabGroupProps


@dataclasses.dataclass
class GuiModalMessage(Message):
    order: float
    uuid: str
    title: str


@dataclasses.dataclass
class GuiCloseModalMessage(Message):
    uuid: str


@dataclasses.dataclass
class GuiButtonProps(GuiBaseProps):
    color: Optional[Color]
    """Color of the button. Synchronized automatically when assigned."""
    _icon_html: Optional[str]
    """(Private) HTML string for the icon to be displayed on the button. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiButtonMessage(Message, tag="GuiComponentMessage"):
    value: bool
    uuid: str
    container_uuid: str
    props: GuiButtonProps


@dataclasses.dataclass
class GuiUploadButtonProps(GuiBaseProps):
    color: Optional[Color]
    """Color of the upload button. Synchronized automatically when assigned."""
    _icon_html: Optional[str]
    """(Private) HTML string for the icon to be displayed on the upload button. Synchronized automatically when assigned."""
    mime_type: str
    """MIME type of the files that can be uploaded. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiUploadButtonMessage(Message, tag="GuiComponentMessage"):
    uuid: str
    container_uuid: str
    props: GuiUploadButtonProps


@dataclasses.dataclass
class GuiSliderProps(GuiBaseProps):
    min: float
    """Minimum value for the slider. Synchronized automatically when assigned."""
    max: float
    """Maximum value for the slider. Synchronized automatically when assigned."""
    step: float
    """Step size for the slider. Synchronized automatically when assigned."""
    precision: int
    """Number of decimal places to display for the slider value. Synchronized automatically when assigned."""
    _marks: Optional[Tuple[GuiSliderMark, ...]] = None
    """(Private) Optional tuple of GuiSliderMark objects to display custom marks on the slider. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiSliderMessage(Message, tag="GuiComponentMessage"):
    value: float
    uuid: str
    container_uuid: str
    props: GuiSliderProps


@dataclasses.dataclass
class GuiMultiSliderProps(GuiBaseProps):
    min: float
    """Minimum value for the multi-slider. Synchronized automatically when assigned."""
    max: float
    """Maximum value for the multi-slider. Synchronized automatically when assigned."""
    step: float
    """Step size for the multi-slider. Synchronized automatically when assigned."""
    min_range: Optional[float]
    """Minimum allowed range between slider handles. Synchronized automatically when assigned."""
    precision: int
    """Number of decimal places to display for the multi-slider values. Synchronized automatically when assigned."""
    fixed_endpoints: bool = False
    """If True, the first and last handles cannot be moved. Synchronized automatically when assigned."""
    _marks: Optional[Tuple[GuiSliderMark, ...]] = None
    """(Private) Optional tuple of GuiSliderMark objects to display custom marks on the multi-slider. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiMultiSliderMessage(Message, tag="GuiComponentMessage"):
    value: tuple[float, ...]
    uuid: str
    container_uuid: str
    props: GuiMultiSliderProps


@dataclasses.dataclass
class GuiNumberProps(GuiBaseProps):
    precision: int
    """Number of decimal places to display for the number value. Synchronized automatically when assigned."""
    step: float
    """Step size for incrementing/decrementing the number value. Synchronized automatically when assigned."""
    min: Optional[float]
    """Minimum allowed value for the number input. Synchronized automatically when assigned."""
    max: Optional[float]
    """Maximum allowed value for the number input. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiNumberMessage(Message, tag="GuiComponentMessage"):
    value: float
    uuid: str
    container_uuid: str
    props: GuiNumberProps


@dataclasses.dataclass
class GuiRgbProps(GuiBaseProps):
    pass


@dataclasses.dataclass
class GuiRgbMessage(Message, tag="GuiComponentMessage"):
    value: Tuple[int, int, int]
    uuid: str
    container_uuid: str
    props: GuiRgbProps


@dataclasses.dataclass
class GuiRgbaProps(GuiBaseProps):
    pass


@dataclasses.dataclass
class GuiRgbaMessage(Message, tag="GuiComponentMessage"):
    value: Tuple[int, int, int, int]
    uuid: str
    container_uuid: str
    props: GuiRgbaProps


@dataclasses.dataclass
class GuiCheckboxProps(GuiBaseProps):
    pass


@dataclasses.dataclass
class GuiCheckboxMessage(Message, tag="GuiComponentMessage"):
    value: bool
    uuid: str
    container_uuid: str
    props: GuiCheckboxProps


@dataclasses.dataclass
class GuiVector2Props(GuiBaseProps):
    min: Optional[Tuple[float, float]]
    """Minimum allowed values for each component of the vector. Synchronized automatically when assigned."""
    max: Optional[Tuple[float, float]]
    """Maximum allowed values for each component of the vector. Synchronized automatically when assigned."""
    step: float
    """Step size for incrementing/decrementing each component of the vector. Synchronized automatically when assigned."""
    precision: int
    """Number of decimal places to display for each component of the vector. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiVector2Message(Message, tag="GuiComponentMessage"):
    value: Tuple[float, float]
    uuid: str
    container_uuid: str
    props: GuiVector2Props


@dataclasses.dataclass
class GuiVector3Props(GuiBaseProps):
    min: Optional[Tuple[float, float, float]]
    """Minimum allowed values for each component of the vector. Synchronized automatically when assigned."""
    max: Optional[Tuple[float, float, float]]
    """Maximum allowed values for each component of the vector. Synchronized automatically when assigned."""
    step: float
    """Step size for incrementing/decrementing each component of the vector. Synchronized automatically when assigned."""
    precision: int
    """Number of decimal places to display for each component of the vector. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiVector3Message(Message, tag="GuiComponentMessage"):
    value: Tuple[float, float, float]
    uuid: str
    container_uuid: str
    props: GuiVector3Props


@dataclasses.dataclass
class GuiTextProps(GuiBaseProps):
    pass


@dataclasses.dataclass
class GuiTextMessage(Message, tag="GuiComponentMessage"):
    value: str
    uuid: str
    container_uuid: str
    props: GuiTextProps


@dataclasses.dataclass
class GuiDropdownProps(GuiBaseProps):
    # This will actually be manually overridden for better types.
    options: Tuple[str, ...]
    """Tuple of options for the dropdown. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiDropdownMessage(Message, tag="GuiComponentMessage"):
    value: str
    uuid: str
    container_uuid: str
    props: GuiDropdownProps


@dataclasses.dataclass
class GuiButtonGroupProps(GuiBaseProps):
    options: Tuple[str, ...]
    """Tuple of buttons for the button group. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GuiButtonGroupMessage(Message, tag="GuiComponentMessage"):
    value: str
    uuid: str
    container_uuid: str
    props: GuiButtonGroupProps


@dataclasses.dataclass
class GuiRemoveMessage(Message):
    """Sent server->client to remove a GUI element."""

    uuid: str


@dataclasses.dataclass
class GuiUpdateMessage(Message):
    """Sent client<->server when any property of a GUI component is changed."""

    uuid: str
    updates: Dict[str, Any]
    """Mapping from property name to new value."""

    @override
    def redundancy_key(self) -> str:
        return (
            type(self).__name__
            + "-"
            + self.uuid
            + "-"
            + ",".join(list(self.updates.keys()))
        )


@dataclasses.dataclass
class SceneNodeUpdateMessage(Message):
    """Sent client<->server when any property of a scene node is changed."""

    name: str
    updates: Dict[str, Any]
    """Mapping from property name to new value."""

    @override
    def redundancy_key(self) -> str:
        return (
            type(self).__name__
            + "-"
            + self.name
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
class CatmullRomSplineMessage(Message, tag="SceneNodeMessage"):
    """Message from server->client carrying Catmull-Rom spline information."""

    name: str
    props: CatmullRomSplineProps


@dataclasses.dataclass
class CatmullRomSplineProps:
    positions: Tuple[Tuple[float, float, float], ...]
    """A tuple of 3D positions (x, y, z) defining the spline's path. Synchronized automatically when assigned."""
    curve_type: Literal["centripetal", "chordal", "catmullrom"]
    """Type of the curve ('centripetal', 'chordal', 'catmullrom'). Synchronized automatically when assigned."""
    tension: float
    """Tension of the curve. Affects the tightness of the curve. Synchronized automatically when assigned."""
    closed: bool
    """Boolean indicating if the spline is closed (forms a loop). Synchronized automatically when assigned."""
    line_width: float
    """Width of the spline line. Synchronized automatically when assigned."""
    color: Tuple[int, int, int]
    """Color of the spline as RGB integers. Synchronized automatically when assigned."""
    segments: Optional[int]
    """Number of segments to divide the spline into. Synchronized automatically when assigned."""


@dataclasses.dataclass
class CubicBezierSplineMessage(Message, tag="SceneNodeMessage"):
    """Message from server->client carrying Cubic Bezier spline information."""

    name: str
    props: CubicBezierSplineProps


@dataclasses.dataclass
class CubicBezierSplineProps:
    positions: Tuple[Tuple[float, float, float], ...]
    """A tuple of 3D positions (x, y, z) defining the spline's key points. Synchronized automatically when assigned."""
    control_points: Tuple[Tuple[float, float, float], ...]
    """A tuple of control points for Bezier curve shaping. Synchronized automatically when assigned."""
    line_width: float
    """Width of the spline line. Synchronized automatically when assigned."""
    color: Tuple[int, int, int]
    """Color of the spline as RGB integers. Synchronized automatically when assigned."""
    segments: Optional[int]
    """Number of segments to divide the spline into. Synchronized automatically when assigned."""


@dataclasses.dataclass
class GaussianSplatsMessage(Message, tag="SceneNodeMessage"):
    """Message from server->client carrying splattable Gaussians."""

    name: str
    props: GaussianSplatsProps


@dataclasses.dataclass
class GaussianSplatsProps:
    # Memory layout is borrowed from:
    # https://github.com/antimatter15/splat
    buffer: npt.NDArray[np.uint32]
    """Our buffer will contain:
    - x as f32
    - y as f32
    - z as f32
    - (unused)
    - chol1 (f16), chol2 (f16)
    - chol3 (f16), chol4 (f16)
    - chol5 (f16), chol6 (f16)
    - rgba (int32)

    Where chol1-6 are the terms of the upper-triangular Cholesky
    factor of covariance matrices."""


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

    source_component_uuid: Optional[str]
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
    source_component_uuid: Optional[str]
    transfer_uuid: str
    part: int
    content: bytes

    @override
    def redundancy_key(self) -> str:
        return type(self).__name__ + "-" + self.transfer_uuid + "-" + str(self.part)


@dataclasses.dataclass
class FileTransferPartAck(Message):
    """Send a file for clients to download or upload files from client."""

    source_component_uuid: Optional[str]
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
