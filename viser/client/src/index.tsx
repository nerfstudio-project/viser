import styled from "@emotion/styled";
import { CameraControls, Environment } from "@react-three/drei";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";

import { SynchronizedCameraControls } from "./CameraControls";
import React from "react";
import { createRoot } from "react-dom/client";

import ControlPanel from "./ControlPanel/ControlPanel";
import {
  SceneNodeThreeObject,
  UseSceneTree,
  useSceneTreeState,
} from "./SceneTree";

import "./index.css";

import Box from "@mui/material/Box";
import WebsocketInterface from "./WebsocketInterface";
import { UseGui, useGuiState } from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";

type ViewerContextContents = {
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  websocketRef: React.MutableRefObject<WebSocket | null>;
  wrapperRef: React.RefObject<HTMLDivElement>;
  objFromSceneNodeNameRef: React.MutableRefObject<{
    [name: string]: THREE.Object3D | undefined;
  }>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
};
export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null
);

function SingleViewer() {
  // Layout and styles.
  const Wrapper = styled(Box)`
    width: 100%;
    height: 100%;
    position: relative;
  `;

  // Default server logic.
  function getDefaultServerFromUrl() {
    // https://localhost:8080/ => ws://localhost:8080
    // https://localhost:8080/?server=some_url => ws://localhost:8080
    let server = window.location.href;
    server = server.replace("http://", "ws://");
    server = server.split("?")[0];
    if (server.endsWith("/")) server = server.slice(0, -1);
    return server;
  }
  const servers = new URLSearchParams(window.location.search).getAll(
    searchParamKey
  );
  const initialServer =
    servers.length >= 1 ? servers[0] : getDefaultServerFromUrl();

  // Values that can be globally accessed by components in a viewer.
  const viewer: ViewerContextContents = {
    useSceneTree: useSceneTreeState(),
    useGui: useGuiState(initialServer),
    websocketRef: React.useRef(null),
    wrapperRef: React.useRef(null),
    objFromSceneNodeNameRef: React.useRef({}),
    sceneRef: React.useRef(null),
    cameraRef: React.useRef(null),
    cameraControlRef: React.useRef(null),
  };
  return (
    <ViewerContext.Provider value={viewer}>
      <Wrapper ref={viewer.wrapperRef}>
        <WebsocketInterface />
        <ControlPanel />
        <ViewerCanvas />
      </Wrapper>
    </ViewerContext.Provider>
  );
}

function ViewerCanvas() {
  const viewer = React.useContext(ViewerContext)!;
  const canvas_background_color = viewer.useGui(
    (state) => state.theme.canvas_background_color
  );
  return (
    <Canvas
      camera={{ position: [3.0, 3.0, -3.0] }}
      style={{
        backgroundColor:
          // Convert int color to hex.
          "#" + canvas_background_color.toString(16).padStart(6, "0"),
        position: "relative",
        zIndex: 0,
        width: "100%",
        height: "100%",
      }}
    >
      <SceneContextSetter />
      <SynchronizedCameraControls />
      <SceneNodeThreeObject name="" useSceneTree={viewer.useSceneTree} />
      <Environment path="/hdri/" files="potsdamer_platz_1k.hdr" />
    </Canvas>
  );
}

/** Component for helping us set the scene reference. */
function SceneContextSetter() {
  const { sceneRef, cameraRef } = React.useContext(ViewerContext)!;
  sceneRef.current = useThree((state) => state.scene);
  cameraRef.current = useThree(
    (state) => state.camera as THREE.PerspectiveCamera
  );
  console.log(cameraRef.current);
  return <></>;
}

function Root() {
  return (
    <Box
      component="div"
      sx={{
        width: "100%",
        height: "100%",
        position: "relative",
        boxSizing: "border-box",
      }}
    >
      <Box
        component="div"
        sx={{
          width: "100%",
          height: "100%",
          boxSizing: "border-box",
          position: "relative",
        }}
      >
        <SingleViewer />
      </Box>
    </Box>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
