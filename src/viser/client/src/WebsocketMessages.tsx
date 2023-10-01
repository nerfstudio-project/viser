// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.
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
  side: "front" | "back" | "double";
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
export interface SceneNodeClickedMessage {
  type: "SceneNodeClickedMessage";
  name: string;
}
/** Reset scene.
 *
 * (automatically generated)
 */
export interface ResetSceneMessage {
  type: "ResetSceneMessage";
}
/** GuiAddFolderMessage(order: 'float', id: 'str', label: 'str', container_id: 'str')
 *
 * (automatically generated)
 */
export interface GuiAddFolderMessage {
  type: "GuiAddFolderMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
}
/** GuiAddMarkdownMessage(order: 'float', id: 'str', markdown: 'str', container_id: 'str')
 *
 * (automatically generated)
 */
export interface GuiAddMarkdownMessage {
  type: "GuiAddMarkdownMessage";
  order: number;
  id: string;
  markdown: string;
  container_id: string;
}
/** GuiAddTabGroupMessage(order: 'float', id: 'str', container_id: 'str', tab_labels: 'Tuple[str, ...]', tab_icons_base64: 'Tuple[Union[str, None], ...]', tab_container_ids: 'Tuple[str, ...]')
 *
 * (automatically generated)
 */
export interface GuiAddTabGroupMessage {
  type: "GuiAddTabGroupMessage";
  order: number;
  id: string;
  container_id: string;
  tab_labels: string[];
  tab_icons_base64: (string | null)[];
  tab_container_ids: string[];
}
/** Base message type containing fields commonly used by GUI inputs.
 *
 * (automatically generated)
 */
export interface _GuiAddInputBase {
  type: "_GuiAddInputBase";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: any;
}
/** GuiAddButtonMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'bool', color: "Optional[Literal[('dark', 'gray', 'red', 'pink', 'grape', 'violet', 'indigo', 'blue', 'cyan', 'green', 'lime', 'yellow', 'orange', 'teal')]]", icon_base64: 'Optional[str]')
 *
 * (automatically generated)
 */
export interface GuiAddButtonMessage {
  type: "GuiAddButtonMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: boolean;
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
}
/** GuiAddSliderMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'float', min: 'float', max: 'float', step: 'Optional[float]', precision: 'int')
 *
 * (automatically generated)
 */
export interface GuiAddSliderMessage {
  type: "GuiAddSliderMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: number;
  min: number;
  max: number;
  step: number | null;
  precision: number;
}
/** GuiAddNumberMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'float', precision: 'int', step: 'float', min: 'Optional[float]', max: 'Optional[float]')
 *
 * (automatically generated)
 */
export interface GuiAddNumberMessage {
  type: "GuiAddNumberMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: number;
  precision: number;
  step: number;
  min: number | null;
  max: number | null;
}
/** GuiAddRgbMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'Tuple[int, int, int]')
 *
 * (automatically generated)
 */
export interface GuiAddRgbMessage {
  type: "GuiAddRgbMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: [number, number, number];
}
/** GuiAddRgbaMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'Tuple[int, int, int, int]')
 *
 * (automatically generated)
 */
export interface GuiAddRgbaMessage {
  type: "GuiAddRgbaMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: [number, number, number, number];
}
/** GuiAddCheckboxMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddCheckboxMessage {
  type: "GuiAddCheckboxMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: boolean;
}
/** GuiAddVector2Message(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'Tuple[float, float]', min: 'Optional[Tuple[float, float]]', max: 'Optional[Tuple[float, float]]', step: 'float', precision: 'int')
 *
 * (automatically generated)
 */
export interface GuiAddVector2Message {
  type: "GuiAddVector2Message";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: [number, number];
  min: [number, number] | null;
  max: [number, number] | null;
  step: number;
  precision: number;
}
/** GuiAddVector3Message(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'Tuple[float, float, float]', min: 'Optional[Tuple[float, float, float]]', max: 'Optional[Tuple[float, float, float]]', step: 'float', precision: 'int')
 *
 * (automatically generated)
 */
export interface GuiAddVector3Message {
  type: "GuiAddVector3Message";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: [number, number, number];
  min: [number, number, number] | null;
  max: [number, number, number] | null;
  step: number;
  precision: number;
}
/** GuiAddTextMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'str')
 *
 * (automatically generated)
 */
export interface GuiAddTextMessage {
  type: "GuiAddTextMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: string;
}
/** GuiAddDropdownMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'str', options: 'Tuple[str, ...]')
 *
 * (automatically generated)
 */
export interface GuiAddDropdownMessage {
  type: "GuiAddDropdownMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: string;
  options: string[];
}
/** GuiAddButtonGroupMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', initial_value: 'str', options: 'Tuple[str, ...]')
 *
 * (automatically generated)
 */
export interface GuiAddButtonGroupMessage {
  type: "GuiAddButtonGroupMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: string;
  options: string[];
}
/** GuiModalMessage(order: 'float', id: 'str', title: 'str')
 *
 * (automatically generated)
 */
export interface GuiModalMessage {
  type: "GuiModalMessage";
  order: number;
  id: string;
  title: string;
}
/** GuiCloseModalMessage(id: 'str')
 *
 * (automatically generated)
 */
export interface GuiCloseModalMessage {
  type: "GuiCloseModalMessage";
  id: string;
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
  dark_mode: boolean;
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
}
/** Message from server->client carrying splattable Gaussians.
 *
 * (automatically generated)
 */
export interface GaussianSplatsMessage {
  type: "GaussianSplatsMessage";
  name: string;
  centers: Uint8Array;
  rgbs: Uint8Array;
  opacities: Uint8Array;
  covariances_triu: Uint8Array;
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

export type Message =
  | ViewerCameraMessage
  | CameraFrustumMessage
  | GlbMessage
  | FrameMessage
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
  | SceneNodeClickedMessage
  | ResetSceneMessage
  | GuiAddFolderMessage
  | GuiAddMarkdownMessage
  | GuiAddTabGroupMessage
  | _GuiAddInputBase
  | GuiAddButtonMessage
  | GuiAddSliderMessage
  | GuiAddNumberMessage
  | GuiAddRgbMessage
  | GuiAddRgbaMessage
  | GuiAddCheckboxMessage
  | GuiAddVector2Message
  | GuiAddVector3Message
  | GuiAddTextMessage
  | GuiAddDropdownMessage
  | GuiAddButtonGroupMessage
  | GuiModalMessage
  | GuiCloseModalMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetVisibleMessage
  | GuiSetDisabledMessage
  | GuiSetValueMessage
  | ThemeConfigurationMessage
  | CatmullRomSplineMessage
  | CubicBezierSplineMessage
  | GaussianSplatsMessage
  | GetRenderRequestMessage
  | GetRenderResponseMessage;
