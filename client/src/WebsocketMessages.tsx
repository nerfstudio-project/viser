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
  position_f32: ArrayBuffer;
  color_uint8: ArrayBuffer;
  point_size: number;
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

export type Message =
  | ViewerCameraMessage
  | CameraFrustumMessage
  | FrameMessage
  | PointCloudMessage
  | ImageMessage
  | RemoveSceneNodeMessage
  | ResetSceneMessage;
