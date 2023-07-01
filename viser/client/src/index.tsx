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
import { Aside, Box, MantineProvider, ScrollArea, useMantineTheme } from "@mantine/core";
import React from "react";
import { createRoot } from "react-dom/client";
import {
  SceneNodeThreeObject,
  UseSceneTree,
  useSceneTreeState,
} from "./SceneTree";

import "./index.css";

import ControlPanel, { ConnectionStatus } from "./ControlPanel/ControlPanel";
import { UseGui, useGuiState } from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";
import WebsocketInterface from "./WebsocketInterface";

import FloatingPanel from "./ControlPanel/FloatingPanel";
import { Titlebar } from "./Titlebar";
import { useMediaQuery } from "@mantine/hooks";
import BottomPanel from "./ControlPanel/BottomPanel";

type ViewerContextContents = {
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  websocketRef: React.MutableRefObject<WebSocket | null>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
  nodeAttributesFromName: React.MutableRefObject<{
    [name: string]:
    | undefined
    | {
      wxyz?: [number, number, number, number];
      position?: [number, number, number];
      visibility?: boolean;
    };
  }>;
};
export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null
);

THREE.ColorManagement.enabled = true;

function SingleViewer() {
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
    canvasRef: React.useRef(null),
    sceneRef: React.useRef(null),
    cameraRef: React.useRef(null),
    cameraControlRef: React.useRef(null),
    // Scene node attributes that aren't placed in the zustand state, for performance reasons.
    nodeAttributesFromName: React.useRef({}),
  };
  const fixed_sidebar = viewer.useGui((state) => state.theme.fixed_sidebar);
  const theme = useMantineTheme();
  const smQuery = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
  return (
    <ViewerContext.Provider value={viewer}>
      <Titlebar />
      <Box
        sx={{
          width: "100%",
          height: "1px",
          position: "relative",
          flex: "1 0 auto",
        }}
      >
        <WebsocketInterface />
        <Box
          sx={(theme) => ({
            top: 0,
            bottom: 0,
            left: 0,
            right: (fixed_sidebar && !smQuery) ? "20em" : 0,
            position: "absolute",
            backgroundColor:
              theme.colorScheme === "light" ? "#fff" : theme.colors.dark[9],
          })}
        >
          <ViewerCanvas />
        </Box>
        <Panel fixed_sidebar={fixed_sidebar} smQuery={smQuery} />
      </Box>
    </ViewerContext.Provider>
  );
}

function Panel(props: { fixed_sidebar: boolean, smQuery: boolean }) {
  if (props.smQuery) {
    return (
      <BottomPanel>
        <BottomPanel.Handle>
          <ConnectionStatus />
        </BottomPanel.Handle>
        <BottomPanel.Contents>
          <ControlPanel />
        </BottomPanel.Contents>
      </BottomPanel>
    )
  }
  else if (props.fixed_sidebar) {
    return (
      <Aside
        hiddenBreakpoint={"sm"}
        sx={(theme) => ({
          width: "20em",
          boxSizing: "border-box",
          right: 0,
          position: "absolute",
          top: "0em",
          bottom: "0em",
          borderLeft: "1px solid",
          borderColor:
            theme.colorScheme == "light"
              ? theme.colors.gray[4]
              : theme.colors.dark[4],
        })}
      >
        <ScrollArea type="always" sx={{ height: "100%" }}>
          <Box
            p="sm"
            sx={(theme) => ({
              backgroundColor:
                theme.colorScheme == "dark"
                  ? theme.colors.dark[5]
                  : theme.colors.gray[1],
              lineHeight: "1.5em",
              fontWeight: 400,
            })}
          >
            <ConnectionStatus />
          </Box>
          <ControlPanel />
        </ScrollArea>
      </Aside>
    )
  }
  else {
    return (
      <FloatingPanel>
        <FloatingPanel.Handle>
          <ConnectionStatus />
        </FloatingPanel.Handle>
        <FloatingPanel.Contents>
          <ControlPanel />
        </FloatingPanel.Contents>
      </FloatingPanel>
    )
  }
}

function ViewerCanvas() {
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
    (state) => state.camera as THREE.PerspectiveCamera
  );
  return <></>;
}

function Root() {
  return (
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
      theme={{
        colorScheme: "light",
      }}
    >
      <Box
        sx={{
          width: "100%",
          height: "100%",
          position: "relative",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <SingleViewer />
      </Box>
    </MantineProvider>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
