// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.
export type Property<T> = { path: string } | { value: T };

/** Button component
 *
 *
 * (automatically generated)
 */
export interface ButtonProps {
  type: "Button";
  label: string;
  color:
    | "dark"
    | "gray"
    | "red"
    | "pink"
    | "grape"
    | "violet"
    | "indigo"
    | "blue"
    | "cyan"
    | "green"
    | "lime"
    | "yellow"
    | "orange"
    | "teal"
    | null;
  icon_base64: string | null;
  disabled: boolean;
  hint: string | null;
}
/** TextInput(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: str, label: str, hint: Optional[str], disabled: bool = False)
 *
 * (automatically generated)
 */
export interface TextInputProps {
  type: "TextInput";
  value: string;
  label: string;
  hint: string | null;
  disabled: boolean;
}
/** NumberInput(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: float, label: str, hint: Optional[str], disabled: bool = False, step: float, min: Optional[float] = None, max: Optional[float] = None, precision: Optional[int] = None)
 *
 * (automatically generated)
 */
export interface NumberInputProps {
  type: "NumberInput";
  value: number;
  label: string;
  hint: string | null;
  disabled: boolean;
  step: number;
  min: number | null;
  max: number | null;
  precision: number | null;
}
/** Slider(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: float, label: str, hint: Optional[str], disabled: bool = False, min: Optional[float] = None, max: Optional[float] = None, step: Optional[float] = None, precision: Optional[int] = None)
 *
 * (automatically generated)
 */
export interface SliderProps {
  type: "Slider";
  value: number;
  label: string;
  hint: string | null;
  disabled: boolean;
  min: number | null;
  max: number | null;
  step: number | null;
  precision: number | null;
}
/** Checkbox(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: bool, label: str, hint: Optional[str], disabled: bool = False)
 *
 * (automatically generated)
 */
export interface CheckboxProps {
  type: "Checkbox";
  value: boolean;
  label: string;
  hint: string | null;
  disabled: boolean;
}
/** RgbInput(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: Tuple[int, int, int], label: str, hint: Optional[str], disabled: bool = False)
 *
 * (automatically generated)
 */
export interface RgbInputProps {
  type: "RgbInput";
  value: [number, number, number];
  label: string;
  hint: string | null;
  disabled: boolean;
}
/** RgbaInput(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: Tuple[int, int, int, int], label: str, hint: Optional[str], disabled: bool = False)
 *
 * (automatically generated)
 */
export interface RgbaInputProps {
  type: "RgbaInput";
  value: [number, number, number, number];
  label: string;
  hint: string | null;
  disabled: boolean;
}
/** Folder(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, label: str, expand_by_default: bool = True)
 *
 * (automatically generated)
 */
export interface FolderProps {
  type: "Folder";
  label: string;
  expand_by_default: boolean;
}
/** Markdown(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, markdown: str)
 *
 * (automatically generated)
 */
export interface MarkdownProps {
  type: "Markdown";
  markdown: string;
}
/** TabGroup(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, tab_labels: Tuple[str, ...], tab_icons_base64: Tuple[Optional[str], ...], tab_container_ids: Tuple[str, ...])
 *
 * (automatically generated)
 */
export interface TabGroupProps {
  type: "TabGroup";
  tab_labels: string[];
  tab_icons_base64: (string | null)[];
  tab_container_ids: string[];
}
/** Modal(*, order: float = <property object at 0x7fb18dbf9c10>, id: str = <property object at 0x7fb18dbf9bc0>, title: str)
 *
 * (automatically generated)
 */
export interface ModalProps {
  type: "Modal";
  order: number;
  id: string;
  title: string;
}
/** Vector2Input(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: Tuple[float, float], label: str, hint: Optional[str], disabled: bool = False, step: float, min: Optional[Tuple[float, float]] = None, max: Optional[Tuple[float, float]] = None, precision: Optional[int] = None)
 *
 * (automatically generated)
 */
export interface Vector2InputProps {
  type: "Vector2Input";
  value: [number, number];
  label: string;
  hint: string | null;
  disabled: boolean;
  step: number;
  min: [number, number] | null;
  max: [number, number] | null;
  precision: number | null;
}
/** Vector3Input(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: Tuple[float, float, float], label: str, hint: Optional[str], disabled: bool = False, min: Optional[Tuple[float, float, float]], max: Optional[Tuple[float, float, float]], step: float, precision: int)
 *
 * (automatically generated)
 */
export interface Vector3InputProps {
  type: "Vector3Input";
  value: [number, number, number];
  label: string;
  hint: string | null;
  disabled: boolean;
  min: [number, number, number] | null;
  max: [number, number, number] | null;
  step: number;
  precision: number;
}
/** Dropdown(*, order: dataclasses.InitVar[typing.Optional[float]] = <property object at 0x7fb18dbf9c10>, value: Optional[str] = None, label: str, hint: Optional[str], disabled: bool = False, options: Tuple[str, ...])
 *
 * (automatically generated)
 */
export interface DropdownProps {
  type: "Dropdown";
  value: string | null;
  label: string;
  hint: string | null;
  disabled: boolean;
  options: string[];
}

/** Add a GUI component.
 *
 * (automatically generated)
 */
export interface GuiAddComponentMessage {
  type: "GuiAddComponentMessage";
  order: number;
  id: string;
  container_id: string;
  props: AllComponentProps;
}
/** Message for a posed viewer camera.
 * Pose is in the form T_world_camera, OpenCV convention, +Z forward.
 *
 * (automatically generated)
 */
export interface ViewerCameraMessage {
  type: "ViewerCameraMessage";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  aspect: number;
  look_at: [number, number, number];
  up_direction: [number, number, number];
}
/** Message for a raycast-like pointer in the scene.
 * origin is the viewing camera position, in world coordinates.
 * direction is the vector if a ray is projected from the camera through the clicked pixel,
 *
 *
 * (automatically generated)
 */
export interface ScenePointerMessage {
  type: "ScenePointerMessage";
  event_type: "click";
  ray_origin: [number, number, number];
  ray_direction: [number, number, number];
}
/** Message to enable/disable scene click events.
 *
 * (automatically generated)
 */
export interface SceneClickEnableMessage {
  type: "SceneClickEnableMessage";
  enable: boolean;
}
/** Variant of CameraMessage used for visualizing camera frustums.
 *
 * OpenCV convention, +Z forward.
 *
 * (automatically generated)
 */
export interface CameraFrustumMessage {
  type: "CameraFrustumMessage";
  name: string;
  fov: number;
  aspect: number;
  scale: number;
  color: number;
  image_media_type: "image/jpeg" | "image/png" | null;
  image_base64_data: string | null;
}
/** GlTF Message
 *
 * (automatically generated)
 */
export interface GlbMessage {
  type: "GlbMessage";
  name: string;
  glb_data: Uint8Array;
  scale: number;
}
/** Coordinate frame message.
 *
 * Position and orientation should follow a `T_parent_local` convention, which
 * corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`.
 *
 * (automatically generated)
 */
export interface FrameMessage {
  type: "FrameMessage";
  name: string;
  show_axes: boolean;
  axes_length: number;
  axes_radius: number;
}
/** Grid message. Helpful for visualizing things like ground planes.
 *
 * (automatically generated)
 */
export interface GridMessage {
  type: "GridMessage";
  name: string;
  width: number;
  height: number;
  width_segments: number;
  height_segments: number;
  plane: "xz" | "xy" | "yx" | "yz" | "zx" | "zy";
  cell_color: number;
  cell_thickness: number;
  cell_size: number;
  section_color: number;
  section_thickness: number;
  section_size: number;
}
/** Add a 2D label to the scene.
 *
 * (automatically generated)
 */
export interface LabelMessage {
  type: "LabelMessage";
  name: string;
  text: string;
}
/** Add a 3D gui element to the scene.
 *
 * (automatically generated)
 */
export interface Gui3DMessage {
  type: "Gui3DMessage";
  order: number;
  name: string;
  container_id: string;
}
/** Point cloud message.
 *
 * Positions are internally canonicalized to float32, colors to uint8.
 *
 * Float color inputs should be in the range [0,1], int color inputs should be in the
 * range [0,255].
 *
 * (automatically generated)
 */
export interface PointCloudMessage {
  type: "PointCloudMessage";
  name: string;
  points: Uint8Array;
  colors: Uint8Array;
  point_size: number;
}
/** Mesh message.
 *
 * Vertices are internally canonicalized to float32, faces to uint32.
 *
 * (automatically generated)
 */
export interface MeshMessage {
  type: "MeshMessage";
  name: string;
  vertices: Uint8Array;
  faces: Uint8Array;
  color: number | null;
  vertex_colors: Uint8Array | null;
  wireframe: boolean;
  opacity: number | null;
  flat_shading: boolean;
  side: "front" | "back" | "double";
  material: "standard" | "toon3" | "toon5";
}
/** Message for transform gizmos.
 *
 * (automatically generated)
 */
export interface TransformControlsMessage {
  type: "TransformControlsMessage";
  name: string;
  scale: number;
  line_width: number;
  fixed: boolean;
  auto_transform: boolean;
  active_axes: [boolean, boolean, boolean];
  disable_axes: boolean;
  disable_sliders: boolean;
  disable_rotations: boolean;
  translation_limits: [[number, number], [number, number], [number, number]];
  rotation_limits: [[number, number], [number, number], [number, number]];
  depth_test: boolean;
  opacity: number;
}
/** Server -> client message to set the camera's position.
 *
 * (automatically generated)
 */
export interface SetCameraPositionMessage {
  type: "SetCameraPositionMessage";
  position: [number, number, number];
}
/** Server -> client message to set the camera's up direction.
 *
 * (automatically generated)
 */
export interface SetCameraUpDirectionMessage {
  type: "SetCameraUpDirectionMessage";
  position: [number, number, number];
}
/** Server -> client message to set the camera's look-at point.
 *
 * (automatically generated)
 */
export interface SetCameraLookAtMessage {
  type: "SetCameraLookAtMessage";
  look_at: [number, number, number];
}
/** Server -> client message to set the camera's field of view.
 *
 * (automatically generated)
 */
export interface SetCameraFovMessage {
  type: "SetCameraFovMessage";
  fov: number;
}
/** Server -> client message to set a scene node's orientation.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetOrientationMessage {
  type: "SetOrientationMessage";
  name: string;
  wxyz: [number, number, number, number];
}
/** Server -> client message to set a scene node's position.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetPositionMessage {
  type: "SetPositionMessage";
  name: string;
  position: [number, number, number];
}
/** Client -> server message when a transform control is updated.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface TransformControlsUpdateMessage {
  type: "TransformControlsUpdateMessage";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
/** Message for rendering a background image.
 *
 * (automatically generated)
 */
export interface BackgroundImageMessage {
  type: "BackgroundImageMessage";
  media_type: "image/jpeg" | "image/png";
  base64_rgb: string;
  base64_depth: string | null;
}
/** Message for rendering 2D images.
 *
 * (automatically generated)
 */
export interface ImageMessage {
  type: "ImageMessage";
  name: string;
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
  render_width: number;
  render_height: number;
}
/** Remove a particular node from the scene.
 *
 * (automatically generated)
 */
export interface RemoveSceneNodeMessage {
  type: "RemoveSceneNodeMessage";
  name: string;
}
/** Set the visibility of a particular node in the scene.
 *
 * (automatically generated)
 */
export interface SetSceneNodeVisibilityMessage {
  type: "SetSceneNodeVisibilityMessage";
  name: string;
  visible: boolean;
}
/** Set the clickability of a particular node in the scene.
 *
 * (automatically generated)
 */
export interface SetSceneNodeClickableMessage {
  type: "SetSceneNodeClickableMessage";
  name: string;
  clickable: boolean;
}
/** Message for clicked objects.
 *
 * (automatically generated)
 */
export interface SceneNodeClickMessage {
  type: "SceneNodeClickMessage";
  name: string;
  ray_origin: [number, number, number];
  ray_direction: [number, number, number];
}
/** Reset scene.
 *
 * (automatically generated)
 */
export interface ResetSceneMessage {
  type: "ResetSceneMessage";
}
/** Sent server->client to remove a GUI element.
 *
 * (automatically generated)
 */
export interface GuiRemoveMessage {
  type: "GuiRemoveMessage";
  id: string;
}
/** Sent client->server when a GUI input is changed.
 *
 * (automatically generated)
 */
export interface GuiUpdateMessage {
  type: "GuiUpdateMessage";
  id: string;
  value: any;
}
/** Sent client->server when a GUI input is changed.
 *
 * (automatically generated)
 */
export interface GuiSetVisibleMessage {
  type: "GuiSetVisibleMessage";
  id: string;
  visible: boolean;
}
/** Sent client->server when a GUI input is changed.
 *
 * (automatically generated)
 */
export interface GuiSetDisabledMessage {
  type: "GuiSetDisabledMessage";
  id: string;
  disabled: boolean;
}
/** Sent server->client to set the value of a particular input.
 *
 * (automatically generated)
 */
export interface GuiSetValueMessage {
  type: "GuiSetValueMessage";
  id: string;
  value: any;
}
/** Message from server->client to configure parts of the GUI.
 *
 * (automatically generated)
 */
export interface ThemeConfigurationMessage {
  type: "ThemeConfigurationMessage";
  titlebar_content: {
    buttons:
      | {
          text: string | null;
          icon: "GitHub" | "Description" | "Keyboard" | null;
          href: string | null;
        }[]
      | null;
    image: {
      image_url_light: string;
      image_url_dark: string | null;
      image_alt: string;
      href: string | null;
    } | null;
  } | null;
  control_layout: "floating" | "collapsible" | "fixed";
  control_width: "small" | "medium" | "large";
  show_logo: boolean;
  dark_mode: boolean;
  colors:
    | [
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
      ]
    | null;
}
/** Message from server->client carrying Catmull-Rom spline information.
 *
 * (automatically generated)
 */
export interface CatmullRomSplineMessage {
  type: "CatmullRomSplineMessage";
  name: string;
  positions: [number, number, number][];
  curve_type: "centripetal" | "chordal" | "catmullrom";
  tension: number;
  closed: boolean;
  line_width: number;
  color: number;
  segments: number | null;
}
/** Message from server->client carrying Cubic Bezier spline information.
 *
 * (automatically generated)
 */
export interface CubicBezierSplineMessage {
  type: "CubicBezierSplineMessage";
  name: string;
  positions: [number, number, number][];
  control_points: [number, number, number][];
  line_width: number;
  color: number;
  segments: number | null;
}
/** Message from server->client requesting a render of the current viewport.
 *
 * (automatically generated)
 */
export interface GetRenderRequestMessage {
  type: "GetRenderRequestMessage";
  format: "image/jpeg" | "image/png";
  height: number;
  width: number;
  quality: number;
}
/** Message from client->server carrying a render.
 *
 * (automatically generated)
 */
export interface GetRenderResponseMessage {
  type: "GetRenderResponseMessage";
  payload: Uint8Array;
}
/** Signal that a file is about to be sent.
 *
 * (automatically generated)
 */
export interface FileDownloadStart {
  type: "FileDownloadStart";
  download_uuid: string;
  filename: string;
  mime_type: string;
  part_count: number;
  size_bytes: number;
}
/** Send a file for clients to download.
 *
 * (automatically generated)
 */
export interface FileDownloadPart {
  type: "FileDownloadPart";
  download_uuid: string;
  part: number;
  content: Uint8Array;
}

export type AllComponentProps =
  | ButtonProps
  | TextInputProps
  | NumberInputProps
  | SliderProps
  | CheckboxProps
  | RgbInputProps
  | RgbaInputProps
  | FolderProps
  | MarkdownProps
  | TabGroupProps
  | ModalProps
  | Vector2InputProps
  | Vector3InputProps
  | DropdownProps;
export type Message =
  | GuiAddComponentMessage
  | ViewerCameraMessage
  | ScenePointerMessage
  | SceneClickEnableMessage
  | CameraFrustumMessage
  | GlbMessage
  | FrameMessage
  | GridMessage
  | LabelMessage
  | Gui3DMessage
  | PointCloudMessage
  | MeshMessage
  | TransformControlsMessage
  | SetCameraPositionMessage
  | SetCameraUpDirectionMessage
  | SetCameraLookAtMessage
  | SetCameraFovMessage
  | SetOrientationMessage
  | SetPositionMessage
  | TransformControlsUpdateMessage
  | BackgroundImageMessage
  | ImageMessage
  | RemoveSceneNodeMessage
  | SetSceneNodeVisibilityMessage
  | SetSceneNodeClickableMessage
  | SceneNodeClickMessage
  | ResetSceneMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetVisibleMessage
  | GuiSetDisabledMessage
  | GuiSetValueMessage
  | ThemeConfigurationMessage
  | CatmullRomSplineMessage
  | CubicBezierSplineMessage
  | GetRenderRequestMessage
  | GetRenderResponseMessage
  | FileDownloadStart
  | FileDownloadPart;
