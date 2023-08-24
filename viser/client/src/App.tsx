// @refresh reset
import {
  AdaptiveDpr,
  AdaptiveEvents,
  CameraControls,
  Environment,
} from "@react-three/drei";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import {
  EffectComposer,
  Outline,
  Selection,
} from "@react-three/postprocessing";
import { BlendFunction, KernelSize } from "postprocessing";

import { SynchronizedCameraControls } from "./CameraControls";
import { Box, MantineProvider, MediaQuery } from "@mantine/core";
import React from "react";
import { SceneNodeThreeObject, UseSceneTree } from "./SceneTree";

import "./index.css";

import ControlPanel from "./ControlPanel/ControlPanel";
import { UseGui, useGuiState, useMantineTheme } from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";
import {
  WebsocketMessageProducer,
  FrameSynchronizedMessageHandler,
} from "./WebsocketInterface";

import { Titlebar } from "./Titlebar";
import { ViserModal } from "./Modal";
import { useSceneTreeState } from "./SceneTreeState";
import { Message } from "./WebsocketMessages";

export type ViewerContextContents = {
  // Zustand hooks.
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  // Useful references.
  websocketRef: React.MutableRefObject<WebSocket | null>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
  // Scene node attributes.
  // This is intentionally placed outside of the Zustand state to reduce overhead.
  nodeAttributesFromName: React.MutableRefObject<{
    [name: string]:
      | undefined
      | {
          wxyz?: [number, number, number, number];
          position?: [number, number, number];
          visibility?: boolean;
        };
  }>;
  messageQueueRef: React.MutableRefObject<Message[]>;
};
export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null,
);

THREE.ColorManagement.enabled = true;

function ViewerRoot() {
  // What websocket server should we connect to?
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
    searchParamKey,
  );
  const initialServer =
    servers.length >= 1 ? servers[0] : getDefaultServerFromUrl();

  // Values that can be globally accessed by components in a viewer.
  const viewer: ViewerContextContents = {
    useSceneTree: useSceneTreeState(),
    useGui: useGuiState(initialServer),
    websocketRef: React.useRef(null),
    canvasRef: React.useRef(null),
    sceneRef: React.useRef(null),
    cameraRef: React.useRef(null),
    cameraControlRef: React.useRef(null),
    // Scene node attributes that aren't placed in the zustand state for performance reasons.
    nodeAttributesFromName: React.useRef({}),
    messageQueueRef: React.useRef([]),
  };

  return (
    <ViewerContext.Provider value={viewer}>
      <WebsocketMessageProducer />
      <ViewerContents />
    </ViewerContext.Provider>
  );
}

function ViewerContents() {
  const viewer = React.useContext(ViewerContext)!;
  const control_layout = viewer.useGui((state) => state.theme.control_layout);
  return (
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
      theme={useMantineTheme()}
    >
      <Titlebar />
      <ViserModal />
      <Box
        sx={{
          width: "100%",
          height: "1px",
          position: "relative",
          flexGrow: 1,
          display: "flex",
          flexDirection: "row",
        }}
      >
        <MediaQuery smallerThan={"xs"} styles={{ right: 0, bottom: "3.5em" }}>
          <Box
            sx={(theme) => ({
              backgroundColor:
                theme.colorScheme === "light" ? "#fff" : theme.colors.dark[9],
              flexGrow: 1,
              width: "10em",
            })}
          >
            <ViewerCanvas>
              <FrameSynchronizedMessageHandler />
            </ViewerCanvas>
          </Box>
        </MediaQuery>
        <ControlPanel control_layout={control_layout} />
      </Box>
    </MantineProvider>
  );
}

function ViewerCanvas({ children }: { children: React.ReactNode }) {
  const viewer = React.useContext(ViewerContext)!;
  return (
    <Canvas
      camera={{ position: [3.0, 3.0, -3.0] }}
      gl={{ preserveDrawingBuffer: true }}
      style={{
        position: "relative",
        zIndex: 0,
        width: "100%",
        height: "100%",
      }}
      performance={{ min: 0.95 }}
      ref={viewer.canvasRef}
    >
      {children}
      <AdaptiveDpr pixelated />
      <AdaptiveEvents />
      <SceneContextSetter />
      <SynchronizedCameraControls />
      <Selection>
        <SceneNodeThreeObject name="" />
        <EffectComposer enabled={true} autoClear={false}>
          <Outline
            hiddenEdgeColor={0xfbff00}
            visibleEdgeColor={0xfbff00}
            blendFunction={BlendFunction.SCREEN} // set this to BlendFunction.ALPHA for dark outlines
            kernelSize={KernelSize.MEDIUM}
            edgeStrength={30.0}
            height={480}
            blur
          />
        </EffectComposer>
      </Selection>
      <Environment path="/hdri/" files="potsdamer_platz_1k.hdr" />
    </Canvas>
  );
}

/** Component for helping us set the scene reference. */
function SceneContextSetter() {
  const { sceneRef, cameraRef } = React.useContext(ViewerContext)!;
  sceneRef.current = useThree((state) => state.scene);
  cameraRef.current = useThree(
    (state) => state.camera as THREE.PerspectiveCamera,
  );
  return <></>;
}

export function Root() {
  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        position: "relative",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <ViewerRoot />
    </Box>
  );
}
