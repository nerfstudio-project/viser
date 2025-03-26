import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./App.css";

import { CameraControls } from "@react-three/drei";
import * as THREE from "three";
import React from "react";
import { UseSceneTree } from "./SceneTree";

import "./index.css";

import { UseGui } from "./ControlPanel/GuiState";
import { GetRenderRequestMessage, Message } from "./WebsocketMessages";

export type ViewerContextContents = {
  messageSource: "websocket" | "file_playback";
  // Zustand hooks.
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  // Useful references.
  // TODO: there's really no reason these all need to be their own ref objects.
  // We could have just one ref to a global mutable struct.
  sendMessageRef: React.MutableRefObject<(message: Message) => void>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  backgroundMaterialRef: React.MutableRefObject<THREE.ShaderMaterial | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
  sendCameraRef: React.MutableRefObject<(() => void) | null>;
  resetCameraViewRef: React.MutableRefObject<(() => void) | null>;
  // Scene node attributes.
  // This is intentionally placed outside of the Zustand state to reduce overhead.
  nodeAttributesFromName: React.MutableRefObject<{
    [name: string]:
      | undefined
      | {
          poseUpdateState?: "updated" | "needsUpdate" | "waitForMakeObject";
          wxyz?: [number, number, number, number];
          position?: [number, number, number];
          visibility?: boolean; // Visibility state from the server.
          overrideVisibility?: boolean; // Override from the GUI.
        };
  }>;
  nodeRefFromName: React.MutableRefObject<{
    [name: string]: undefined | THREE.Object3D;
  }>;
  messageQueueRef: React.MutableRefObject<Message[]>;
  // Requested a render.
  getRenderRequestState: React.MutableRefObject<
    "ready" | "triggered" | "pause" | "in_progress"
  >;
  getRenderRequest: React.MutableRefObject<null | GetRenderRequestMessage>;
  // Track click drag events.
  scenePointerInfo: React.MutableRefObject<{
    enabled: false | "click" | "rect-select"; // Enable box events.
    dragStart: [number, number]; // First mouse position.
    dragEnd: [number, number]; // Final mouse position.
    isDragging: boolean;
  }>;
  // 2D canvas for drawing -- can be used to give feedback on cursor movement, or more.
  canvas2dRef: React.MutableRefObject<HTMLCanvasElement | null>;
  // Poses for bones in skinned meshes.
  skinnedMeshState: React.MutableRefObject<{
    [name: string]: {
      initialized: boolean;
      poses: {
        wxyz: [number, number, number, number];
        position: [number, number, number];
      }[];
    };
  }>;

  // Global hover state tracking for cursor management
  hoveredElementsCount: React.MutableRefObject<number>;
};
export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null,
);
