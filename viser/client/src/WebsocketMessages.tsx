// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.

// For numpy arrays, we directly serialize the underlying data buffer.
type ArrayBuffer = Uint8Array;
export interface ViewerCameraMessage {
  type: "ViewerCameraMessage";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  aspect: number;
  look_at: [number, number, number];
  up_direction: [number, number, number];
}
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
export interface FrameMessage {
  type: "FrameMessage";
  name: string;
  show_axes: boolean;
  axes_length: number;
  axes_radius: number;
}
export interface LabelMessage {
  type: "LabelMessage";
  name: string;
  text: string;
}
export interface PointCloudMessage {
  type: "PointCloudMessage";
  name: string;
  points: ArrayBuffer;
  colors: ArrayBuffer;
  point_size: number;
}
export interface MeshMessage {
  type: "MeshMessage";
  name: string;
  vertices: ArrayBuffer;
  faces: ArrayBuffer;
  color: number | null;
  vertex_colors: ArrayBuffer | null;
  wireframe: boolean;
  side: "front" | "back" | "double";
}
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
export interface SetCameraPositionMessage {
  type: "SetCameraPositionMessage";
  position: [number, number, number];
}
export interface SetCameraUpDirectionMessage {
  type: "SetCameraUpDirectionMessage";
  position: [number, number, number];
}
export interface SetCameraLookAtMessage {
  type: "SetCameraLookAtMessage";
  look_at: [number, number, number];
}
export interface SetCameraFovMessage {
  type: "SetCameraFovMessage";
  fov: number;
}
export interface SetOrientationMessage {
  type: "SetOrientationMessage";
  name: string;
  wxyz: [number, number, number, number];
}
export interface SetPositionMessage {
  type: "SetPositionMessage";
  name: string;
  position: [number, number, number];
}
export interface TransformControlsUpdateMessage {
  type: "TransformControlsUpdateMessage";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
export interface BackgroundImageMessage {
  type: "BackgroundImageMessage";
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
}
export interface ImageMessage {
  type: "ImageMessage";
  name: string;
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
  render_width: number;
  render_height: number;
}
export interface RemoveSceneNodeMessage {
  type: "RemoveSceneNodeMessage";
  name: string;
}
export interface SetSceneNodeVisibilityMessage {
  type: "SetSceneNodeVisibilityMessage";
  name: string;
  visible: boolean;
}
export interface SetSceneNodeClickableMessage {
  type: "SetSceneNodeClickableMessage";
  name: string;
  clickable: boolean;
}
export interface SceneNodeClickedMessage {
  type: "SceneNodeClickedMessage";
  name: string;
}
export interface ResetSceneMessage {
  type: "ResetSceneMessage";
}
export interface GuiAddFolderMessage {
  type: "GuiAddFolderMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
}
export interface GuiAddTabGroupMessage {
  type: "GuiAddTabGroupMessage";
  order: number;
  id: string;
  container_id: string;
  tab_labels: string[];
  tab_icons_base64: (string | null)[];
  tab_container_ids: string[];
}
export interface _GuiAddInputBase {
  type: "_GuiAddInputBase";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: any;
}
export interface GuiAddButtonMessage {
  type: "GuiAddButtonMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: boolean;
}
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
export interface GuiAddRgbMessage {
  type: "GuiAddRgbMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: [number, number, number];
}
export interface GuiAddRgbaMessage {
  type: "GuiAddRgbaMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: [number, number, number, number];
}
export interface GuiAddCheckboxMessage {
  type: "GuiAddCheckboxMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: boolean;
}
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
export interface GuiAddTextMessage {
  type: "GuiAddTextMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  initial_value: string;
}
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
export interface GuiRemoveContainerChildrenMessage {
  type: "GuiRemoveContainerChildrenMessage";
  container_id: string;
}
export interface GuiRemoveMessage {
  type: "GuiRemoveMessage";
  id: string;
}
export interface GuiUpdateMessage {
  type: "GuiUpdateMessage";
  id: string;
  value: any;
}
export interface GuiSetVisibleMessage {
  type: "GuiSetVisibleMessage";
  id: string;
  visible: boolean;
}
export interface GuiSetDisabledMessage {
  type: "GuiSetDisabledMessage";
  id: string;
  disabled: boolean;
}
export interface GuiSetValueMessage {
  type: "GuiSetValueMessage";
  id: string;
  value: any;
}
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
  dark_mode: boolean;
}

export type Message =
  | ViewerCameraMessage
  | CameraFrustumMessage
  | FrameMessage
  | LabelMessage
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
  | GuiRemoveContainerChildrenMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetVisibleMessage
  | GuiSetDisabledMessage
  | GuiSetValueMessage
  | ThemeConfigurationMessage;
