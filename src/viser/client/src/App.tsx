// @refresh reset
import {
  AdaptiveDpr,
  AdaptiveEvents,
  CameraControls,
  Environment,
} from "@react-three/drei";
import * as THREE from "three";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import {
  EffectComposer,
  Outline,
  Selection,
} from "@react-three/postprocessing";
import { BlendFunction, KernelSize } from "postprocessing";

import { SynchronizedCameraControls } from "./CameraControls";
import { Box, Image, MantineProvider, MediaQuery } from "@mantine/core";
import React from "react";
import { SceneNodeThreeObject, UseSceneTree } from "./SceneTree";

import "./index.css";

import ControlPanel from "./ControlPanel/ControlPanel";
import {
  UseGui,
  useGuiState,
  useViserMantineTheme,
} from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";
import {
  WebsocketMessageProducer,
  FrameSynchronizedMessageHandler,
} from "./WebsocketInterface";

import { Titlebar } from "./Titlebar";
import { ViserModal } from "./Modal";
import { useSceneTreeState } from "./SceneTreeState";
import { GetRenderRequestMessage, Message } from "./WebsocketMessages";
import { makeThrottledMessageSender } from "./WebsocketFunctions";

export type ViewerContextContents = {
  // Zustand hooks.
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  // Useful references.
  websocketRef: React.MutableRefObject<WebSocket | null>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  backgroundMaterialRef: React.MutableRefObject<THREE.ShaderMaterial | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
  resetCameraViewRef: React.MutableRefObject<(() => void) | null>;
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
  // Requested a render.
  getRenderRequestState: React.MutableRefObject<
    "ready" | "triggered" | "pause" | "in_progress"
  >;
  getRenderRequest: React.MutableRefObject<null | GetRenderRequestMessage>;
  sceneClickEnable: React.MutableRefObject<boolean>;
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
    server = server.replace("https://", "wss://");
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
    backgroundMaterialRef: React.useRef(null),
    cameraControlRef: React.useRef(null),
    resetCameraViewRef: React.useRef(null),
    // Scene node attributes that aren't placed in the zustand state for performance reasons.
    nodeAttributesFromName: React.useRef({}),
    messageQueueRef: React.useRef([]),
    getRenderRequestState: React.useRef("ready"),
    getRenderRequest: React.useRef(null),
    sceneClickEnable: React.useRef(false),
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
      theme={useViserMantineTheme()}
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
              position: "relative",
            })}
          >
            <ViewerCanvas>
              <FrameSynchronizedMessageHandler />
            </ViewerCanvas>
            {viewer.useGui((state) => state.theme.show_logo) ? (
              <Box
                sx={{
                  position: "absolute",
                  bottom: "1em",
                  left: "1em",
                  filter: "saturate(0.625)",
                  "&:hover": {
                    filter: "saturate(1.0)",
                  },
                }}
                component="a"
                target="_blank"
                href="https://viser.studio"
              >
                <Image src="/logo.svg" width="2.5em" height="auto" />
              </Box>
            ) : null}
          </Box>
        </MediaQuery>
        <ControlPanel control_layout={control_layout} />
      </Box>
    </MantineProvider>
  );
}

function ViewerCanvas({ children }: { children: React.ReactNode }) {
  const viewer = React.useContext(ViewerContext)!;
  const sendClickThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    20,
  );
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
      // Handle scene click events.
      onClick={(e) => {
        // Don't send click events if the scene pointer events are disabled.
        if (!viewer.sceneClickEnable.current) return;

        // clientX/Y are relative to the viewport, offsetX/Y are relative to the canvasRef.
        // clientX==offsetX if there is no titlebar, but clientX>offsetX if there is a titlebar.
        const mouseVector = new THREE.Vector2();
        mouseVector.x =
          2 * (e.nativeEvent.offsetX / viewer.canvasRef.current!.clientWidth) -
          1;
        mouseVector.y =
          1 -
          2 * (e.nativeEvent.offsetY / viewer.canvasRef.current!.clientHeight);
        if (
          mouseVector.x > 1 ||
          mouseVector.x < -1 ||
          mouseVector.y > 1 ||
          mouseVector.y < -1
        )
          return;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouseVector, viewer.cameraRef.current!);

        sendClickThrottled({
          type: "ScenePointerMessage",
          event_type: "click",
          ray_origin: [
            raycaster.ray.origin.x,
            -raycaster.ray.origin.z,
            raycaster.ray.origin.y,
          ],
          ray_direction: [
            raycaster.ray.direction.x,
            -raycaster.ray.direction.z,
            raycaster.ray.direction.y,
          ],
        });
      }}
    >
      {children}
      <BackgroundImage />
      <AdaptiveDpr pixelated />
      <AdaptiveEvents />
      <SceneContextSetter />
      <SynchronizedCameraControls />
      <Selection>
        <SceneNodeThreeObject name="" parent={null} />
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

/* Background image with support for depth compositing. */
function BackgroundImage() {
  // Create a fragment shader that composites depth using depth and rgb
  const vertShader = `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
  `.trim();
  const fragShader = `  
  #include <packing>
  precision highp float;
  precision highp int;

  varying vec2 vUv;
  uniform sampler2D colorMap;
  uniform sampler2D depthMap;
  uniform float cameraNear;
  uniform float cameraFar;
  uniform bool enabled;
  uniform bool hasDepth;

  float readDepth(sampler2D depthMap, vec2 coord) {
    vec4 rgbPacked = texture(depthMap, coord);

    // For the k-th channel, coefficients are calculated as: 255 * 1e-5 * 2^(8 * k).
    // Note that: [0, 255] channels are scaled to [0, 1], and we multiply by 1e5 on the server side.
    float depth = rgbPacked.r * 0.00255 + rgbPacked.g * 0.6528 + rgbPacked.b * 167.1168;
    return depth;
  }

  void main() {
    if (!enabled) {
      // discard the pixel if we're not enabled
      discard;
    }
    vec4 color = texture(colorMap, vUv);
    gl_FragColor = vec4(color.rgb, 1.0);

    float bufDepth;
    if(hasDepth){
      float depth = readDepth(depthMap, vUv);
      bufDepth = viewZToPerspectiveDepth(-depth, cameraNear, cameraFar);
    } else {
      // If no depth enabled, set depth to 1.0 (infinity) to treat it like a background image.
      bufDepth = 1.0;
    }
    gl_FragDepth = bufDepth;
  }`.trim();
  // initialize the rgb texture with all white and depth at infinity
  const backgroundMaterial = new THREE.ShaderMaterial({
    fragmentShader: fragShader,
    vertexShader: vertShader,
    uniforms: {
      enabled: { value: false },
      depthMap: { value: null },
      colorMap: { value: null },
      cameraNear: { value: null },
      cameraFar: { value: null },
      hasDepth: { value: false },
    },
  });
  const { backgroundMaterialRef } = React.useContext(ViewerContext)!;
  backgroundMaterialRef.current = backgroundMaterial;
  const backgroundMesh = React.useRef<THREE.Mesh>(null);
  useFrame(({ camera }) => {
    // Logic ahead relies on perspective camera assumption.
    if (!(camera instanceof THREE.PerspectiveCamera)) {
      console.error(
        "Camera is not a perspective camera, cannot render background image",
      );
      return;
    }

    // Update the position of the mesh based on the camera position.
    const lookdir = camera.getWorldDirection(new THREE.Vector3());
    backgroundMesh.current!.position.set(
      camera.position.x,
      camera.position.y,
      camera.position.z,
    );
    backgroundMesh.current!.position.addScaledVector(lookdir, 1.0);
    backgroundMesh.current!.quaternion.copy(camera.quaternion);

    // Resize the mesh based on focal length.
    const f = camera.getFocalLength();
    backgroundMesh.current!.scale.set(
      camera.getFilmWidth() / f,
      camera.getFilmHeight() / f,
      1.0,
    );

    // Set near/far uniforms.
    backgroundMaterial.uniforms.cameraNear.value = camera.near;
    backgroundMaterial.uniforms.cameraFar.value = camera.far;
  });

  return (
    <mesh
      ref={backgroundMesh}
      material={backgroundMaterial}
      matrixWorldAutoUpdate={false}
    >
      <planeGeometry attach="geometry" args={[1, 1]} />
    </mesh>
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
