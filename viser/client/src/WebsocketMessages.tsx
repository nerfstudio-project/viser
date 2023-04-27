// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.

// For numpy arrays, we directly serialize the underlying data buffer.
type ArrayBuffer = Uint8Array;
interface ViewerCameraMessage {
  type: "ViewerCameraMessage";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  aspect: number;
  look_at: [number, number, number];
  up_direction: [number, number, number];
}
interface CameraFrustumMessage {
  type: "CameraFrustumMessage";
  name: string;
  fov: number;
  aspect: number;
  scale: number;
  color: number;
}
interface FrameMessage {
  type: "FrameMessage";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
  show_axes: boolean;
  axes_length: number;
  axes_radius: number;
}
interface LabelMessage {
  type: "LabelMessage";
  name: string;
  text: string;
}
interface PointCloudMessage {
  type: "PointCloudMessage";
  name: string;
  position: ArrayBuffer;
  color: ArrayBuffer;
  point_size: number;
}
interface MeshMessage {
  type: "MeshMessage";
  name: string;
  vertices: ArrayBuffer;
  faces: ArrayBuffer;
  color: number;
  wireframe: boolean;
}
interface TransformControlsMessage {
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
interface SetCameraPositionMessage {
  type: "SetCameraPositionMessage";
  position: [number, number, number];
}
interface SetCameraUpDirectionMessage {
  type: "SetCameraUpDirectionMessage";
  position: [number, number, number];
}
interface SetCameraLookAtMessage {
  type: "SetCameraLookAtMessage";
  look_at: [number, number, number];
}
interface SetCameraFovMessage {
  type: "SetCameraFovMessage";
  fov: number;
}
interface SetOrientationMessage {
  type: "SetOrientationMessage";
  name: string;
  wxyz: [number, number, number, number];
}
interface SetPositionMessage {
  type: "SetPositionMessage";
  name: string;
  position: [number, number, number];
}
interface TransformControlsUpdateMessage {
  type: "TransformControlsUpdateMessage";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
interface BackgroundImageMessage {
  type: "BackgroundImageMessage";
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
}
interface ImageMessage {
  type: "ImageMessage";
  name: string;
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
  render_width: number;
  render_height: number;
}
interface RemoveSceneNodeMessage {
  type: "RemoveSceneNodeMessage";
  name: string;
}
interface SetSceneNodeVisibilityMessage {
  type: "SetSceneNodeVisibilityMessage";
  name: string;
  visible: boolean;
}
interface ResetSceneMessage {
  type: "ResetSceneMessage";
}
interface GuiAddMessage {
  type: "GuiAddMessage";
  name: string;
  folder_labels: string[];
  leva_conf: any;
}
interface GuiRemoveMessage {
  type: "GuiRemoveMessage";
  name: string;
}
interface GuiUpdateMessage {
  type: "GuiUpdateMessage";
  name: string;
  value: any;
}
interface GuiSetVisibleMessage {
  type: "GuiSetVisibleMessage";
  name: string;
  visible: boolean;
}
interface GuiSetValueMessage {
  type: "GuiSetValueMessage";
  name: string;
  value: any;
}
interface GuiSetLevaConfMessage {
  type: "GuiSetLevaConfMessage";
  name: string;
  leva_conf: any;
}
interface MessageGroupStart {
  type: "MessageGroupStart";
}
interface MessageGroupEnd {
  type: "MessageGroupEnd";
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
  | ResetSceneMessage
  | GuiAddMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetVisibleMessage
  | GuiSetValueMessage
  | GuiSetLevaConfMessage
  | MessageGroupStart
  | MessageGroupEnd;
