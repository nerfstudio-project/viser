// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.

// For numpy arrays, we directly serialize the underlying data buffer.
type ArrayBuffer = Uint8Array;

export interface ViewerCameraMessage {
  type: "viewer_camera";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  aspect: number;
}
export interface CameraFrustumMessage {
  type: "camera_frustum";
  name: string;
  fov: number;
  aspect: number;
  scale: number;
}
export interface FrameMessage {
  type: "frame";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
  show_axes: boolean;
  scale: number;
}
export interface PointCloudMessage {
  type: "point_cloud";
  name: string;
  position: ArrayBuffer;
  color: ArrayBuffer;
  point_size: number;
}
export interface MeshMessage {
  type: "mesh";
  name: string;
  vertices: ArrayBuffer;
  faces: ArrayBuffer;
}
export interface BackgroundImageMessage {
  type: "background_image";
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
}
export interface ImageMessage {
  type: "image";
  name: string;
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
  render_width: number;
  render_height: number;
}
export interface RemoveSceneNodeMessage {
  type: "remove_scene_node";
  name: string;
}
export interface ResetSceneMessage {
  type: "reset_scene";
}
export interface GuiAddMessage {
  type: "add_gui";
  name: string;
  folder: string;
  leva_conf: any;
}
export interface GuiRemoveMessage {
  type: "remove_gui";
  name: string;
}
export interface GuiUpdateMessage {
  type: "gui_update";
  name: string;
  value: any;
}
export interface GuiSetMessage {
  type: "gui_set";
  name: string;
  value: any;
}

export type Message =
  | ViewerCameraMessage
  | CameraFrustumMessage
  | FrameMessage
  | PointCloudMessage
  | MeshMessage
  | BackgroundImageMessage
  | ImageMessage
  | RemoveSceneNodeMessage
  | ResetSceneMessage
  | GuiAddMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetMessage;
