import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./App.css";
import "./index.css";

import { CameraControls } from "@react-three/drei";
import * as THREE from "three";
import React from "react";
import { UseSceneTree } from "./SceneTree";

import { UseGui } from "./ControlPanel/GuiState";
import { GetRenderRequestMessage, Message } from "./WebsocketMessages";

// Type definitions for all mutable references.
export type ViewerRefs = {
  // Function references.
  sendMessage: (message: Message) => void;
  sendCamera: (() => void) | null;
  resetCameraView: (() => void) | null;

  // DOM/Three.js references.
  canvas: HTMLCanvasElement | null;
  canvas2d: HTMLCanvasElement | null;
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  backgroundMaterial: THREE.ShaderMaterial | null;
  cameraControl: CameraControls | null;

  // Scene management.
  nodeAttributesFromName: {
    [name: string]:
      | undefined
      | {
          poseUpdateState?: "updated" | "needsUpdate" | "waitForMakeObject";
          wxyz?: [number, number, number, number];
          position?: [number, number, number];
          visibility?: boolean; // Visibility state from the server.
          overrideVisibility?: boolean; // Override from the GUI.
        };
  };
  nodeRefFromName: {
    [name: string]: undefined | THREE.Object3D;
  };

  // Message and rendering state.
  messageQueue: Message[];
  getRenderRequestState: "ready" | "triggered" | "pause" | "in_progress";
  getRenderRequest: null | GetRenderRequestMessage;

  // Interaction state.
  scenePointerInfo: {
    enabled: false | "click" | "rect-select"; // Enable box events.
    dragStart: [number, number]; // First mouse position.
    dragEnd: [number, number]; // Final mouse position.
    isDragging: boolean;
  };

  // Skinned mesh state.
  skinnedMeshState: {
    [name: string]: {
      initialized: boolean;
      poses: {
        wxyz: [number, number, number, number];
        position: [number, number, number];
      }[];
    };
  };

  // Global hover state tracking.
  hoveredElementsCount: number;
};

export type ViewerContextContents = {
  // Non-mutable state.
  messageSource: "websocket" | "file_playback";

  // Zustand state hooks.
  useSceneTree: UseSceneTree;
  useGui: UseGui;

  // Single reference to all mutable state.
  refs: React.MutableRefObject<ViewerRefs>;
};

export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null,
);
