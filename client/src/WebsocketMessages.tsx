// Message type definitions.
//
// These currently need to be synchronizd manually with the corresponding
// Python definitions.

type ArrayBuffer = Uint8Array;

interface FrameMessage {
  type: "frame";
  name: string;
  xyzw: [number, number, number, number];
  position: [number, number, number];
  show_axes: boolean;
}
interface PointCloudMessage {
  type: "point_cloud";
  name: string;
  position_f32: ArrayBuffer;
  color_uint8: ArrayBuffer;
  point_size: number;
}
interface CameraFrustumMessage {
  type: "camera_frustum";
  name: string;
  fov: number;
  aspect: number;
}
interface RemoveSceneNodeMessage {
  type: "remove_scene_node";
  name: string;
}
interface ResetSceneMessage {
  type: "reset_scene";
}

export type Message =
  | FrameMessage
  | PointCloudMessage
  | CameraFrustumMessage
  | RemoveSceneNodeMessage
  | ResetSceneMessage;
