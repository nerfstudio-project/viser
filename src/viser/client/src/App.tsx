// @refresh reset
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./App.css";
import "./index.css";

import { useInView } from "react-intersection-observer";
import { Notifications } from "@mantine/notifications";
import { Environment, PerformanceMonitor, Stats, Bvh } from "@react-three/drei";
import * as THREE from "three";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import React, { useEffect, useMemo, useState } from "react";
import { ViewerMutable } from "./ViewerContext";
import {
  Anchor,
  Box,
  Image,
  MantineProvider,
  Modal,
  Tooltip,
  createTheme,
  useMantineColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";

// Local imports.
import { SynchronizedCameraControls } from "./CameraControls";
import { SceneNodeThreeObject } from "./SceneTree";
import { ViewerContext, ViewerContextContents } from "./ViewerContext";
import ControlPanel from "./ControlPanel/ControlPanel";
import { useGuiState } from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";
import { WebsocketMessageProducer } from "./WebsocketInterface";
import { Titlebar } from "./Titlebar";
import { ViserModal } from "./Modal";
import { useSceneTreeState } from "./SceneTreeState";
import { useThrottledMessageSender } from "./WebsocketUtils";
import { rayToViserCoords } from "./WorldTransformUtils";
import { theme } from "./AppTheme";
import { FrameSynchronizedMessageHandler } from "./MessageHandler";
import { PlaybackFromFile } from "./FilePlayback";
import { SplatRenderContext } from "./Splatting/GaussianSplats";
import { BrowserWarning } from "./BrowserWarning";
import { MacWindowWrapper } from "./MacWindowWrapper";
import { CsmDirectionalLight } from "./CsmDirectionalLight";
import { VISER_VERSION } from "./VersionInfo";

// ======= Utility functions =======

/** Turn a click event into a normalized device coordinate (NDC) vector.
 * Normalizes click coordinates to be between -1 and 1, with (0, 0) being the center of the screen.
 *
 * Returns null if input is not valid.
 */
function ndcFromPointerXy(
  viewer: ViewerContextContents,
  xy: [number, number],
): THREE.Vector2 | null {
  const mouseVector = new THREE.Vector2();
  mouseVector.x =
    2 * ((xy[0] + 0.5) / viewer.mutable.current.canvas!.clientWidth) - 1;
  mouseVector.y =
    1 - 2 * ((xy[1] + 0.5) / viewer.mutable.current.canvas!.clientHeight);
  return mouseVector.x < 1 &&
    mouseVector.x > -1 &&
    mouseVector.y < 1 &&
    mouseVector.y > -1
    ? mouseVector
    : null;
}

/** Turn a click event to normalized OpenCV coordinate (NDC) vector.
 * Normalizes click coordinates to be between (0, 0) as upper-left corner,
 * and (1, 1) as lower-right corner, with (0.5, 0.5) being the center of the screen.
 * Uses offsetX/Y, and clientWidth/Height to get the coordinates.
 */
function opencvXyFromPointerXy(
  viewer: ViewerContextContents,
  xy: [number, number],
): THREE.Vector2 {
  const mouseVector = new THREE.Vector2();
  mouseVector.x = (xy[0] + 0.5) / viewer.mutable.current.canvas!.clientWidth;
  mouseVector.y = (xy[1] + 0.5) / viewer.mutable.current.canvas!.clientHeight;
  return mouseVector;
}

/** Gets default WebSocket server URL based on current window location. */
const getDefaultServerFromUrl = (): string => {
  let server = window.location.href;
  server = server.replace("http://", "ws://");
  server = server.replace("https://", "wss://");
  server = server.split("?")[0];
  if (server.endsWith("/")) server = server.slice(0, -1);
  return server;
};

/** Disables rendering when component is not in view. */
const DisableRender = (): null => useFrame(() => null, 1000);

// ======= Main component tree =======

/**
 * Root application component - handles dummy window wrapper if needed.
 */
export function Root() {
  const searchParams = new URLSearchParams(window.location.search);
  const dummyWindowParam = searchParams.get("dummyWindowDimensions");
  const dummyWindowTitle =
    searchParams.get("dummyWindowTitle") ?? "localhost:8080";

  const content = (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <ViewerRoot />
    </div>
  );

  // If dummy window dimensions are specified, wrap content in MacWindowWrapper.
  if (!dummyWindowParam) return content;

  const [width, height] = dummyWindowParam.split("x").map(Number);
  if (isNaN(width) || isNaN(height)) return content;

  return (
    <MacWindowWrapper title={dummyWindowTitle} width={width} height={height}>
      {content}
    </MacWindowWrapper>
  );
}

/**
 * Main viewer context provider component.
 */
function ViewerRoot() {
  // Server configuration and URL parameters.
  const servers = new URLSearchParams(window.location.search).getAll(
    searchParamKey,
  );
  const initialServer =
    servers.length >= 1 ? servers[0] : getDefaultServerFromUrl();

  const searchParams = new URLSearchParams(window.location.search);
  const playbackPath = searchParams.get("playbackPath");
  const darkMode = searchParams.get("darkMode") !== null;
  const showStats = searchParams.get("showStats") !== null;

  // Create a message source string.
  const messageSource = playbackPath === null ? "websocket" : "file_playback";

  // Create a default quaternion for the world frame.
  const defaultQuat = (() => {
    const quat = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(Math.PI / 2, Math.PI, -Math.PI / 2),
    );
    return [quat.w, quat.x, quat.y, quat.z] as [number, number, number, number];
  })();

  // Create a single ref with all mutable state.
  const nodeRefFromName = {};
  const mutable = React.useRef<ViewerMutable>({
    // Function references with default implementations.
    sendMessage:
      playbackPath == null
        ? (message: any) =>
            console.log(
              `Tried to send ${message.type} but websocket is not connected!`,
            )
        : () => null,
    sendCamera: null,
    resetCameraView: null,

    // DOM/Three.js references.
    canvas: null,
    canvas2d: null,
    scene: null,
    camera: null,
    backgroundMaterial: null,
    cameraControl: null,

    // Scene management.
    nodeAttributesFromName: {
      "": {
        wxyz: defaultQuat,
      },
    },
    nodeRefFromName,

    // Message and rendering state.
    messageQueue: [],
    getRenderRequestState: "ready",
    getRenderRequest: null,

    // Interaction state.
    scenePointerInfo: {
      enabled: false,
      dragStart: [0, 0],
      dragEnd: [0, 0],
      isDragging: false,
    },

    // Skinned mesh state.
    skinnedMeshState: {},

    // Global hover state tracking.
    hoveredElementsCount: 0,
  });

  // Create the context value with hooks and single ref.
  const viewer: ViewerContextContents = {
    messageSource,
    useSceneTree: useSceneTreeState(mutable.current.nodeRefFromName),
    useGui: useGuiState(initialServer),
    mutable,
  };

  // Apply URL dark mode setting if provided.
  if (darkMode) viewer.useGui.getState().theme.dark_mode = darkMode;

  return (
    <ViewerContext.Provider value={viewer}>
      <ViewerContents>
        {messageSource === "websocket" && <WebsocketMessageProducer />}
        {messageSource === "file_playback" && (
          <PlaybackFromFile fileUrl={playbackPath!} />
        )}
        {showStats && <Stats className="stats-panel" />}
      </ViewerContents>
    </ViewerContext.Provider>
  );
}

/**
 * Main content wrapper with theme and layout.
 */
function ViewerContents({ children }: { children: React.ReactNode }) {
  const viewer = React.useContext(ViewerContext)!;
  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const colors = viewer.useGui((state) => state.theme.colors);
  const controlLayout = viewer.useGui((state) => state.theme.control_layout);
  const showLogo = viewer.useGui((state) => state.theme.show_logo);
  const { messageSource } = viewer;

  // Create Mantine theme with custom colors if provided.
  const mantineTheme = useMemo(
    () =>
      createTheme({
        ...theme,
        ...(colors === null
          ? {}
          : { colors: { custom: colors }, primaryColor: "custom" }),
      }),
    [colors],
  );

  return (
    <>
      <MantineProvider
        theme={mantineTheme}
        defaultColorScheme={darkMode ? "dark" : "light"}
        colorSchemeManager={{
          // Mock external color scheme manager. This prevents multiple Viser
          // instances from affecting each others' color schemes.
          get: (defaultValue) => defaultValue,
          set: () => null,
          subscribe: () => null,
          unsubscribe: () => null,
          clear: () => null,
        }}
      >
        {children}
        <ColorSchemeSetter darkMode={darkMode} />
        <NotificationsPanel />
        <BrowserWarning />
        <ViserModal />
        {/* App layout */}
        <Box
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            position: "relative",
            flexDirection: "column",
          }}
        >
          <Titlebar />
          <Box
            style={{
              width: "100%",
              position: "relative",
              flexGrow: 1,
              overflow: "hidden",
              display: "flex",
            }}
          >
            <Box
              style={(theme) => ({
                backgroundColor: darkMode ? theme.colors.dark[9] : "#fff",
                flexGrow: 1,
                overflow: "hidden",
                height: "100%",
              })}
            >
              <Viewer2DCanvas />
              <ViewerCanvas>
                <FrameSynchronizedMessageHandler />
              </ViewerCanvas>
              {showLogo && messageSource === "websocket" && <ViserLogo />}
            </Box>
            {messageSource === "websocket" && (
              <ControlPanel control_layout={controlLayout} />
            )}
          </Box>
        </Box>
      </MantineProvider>
    </>
  );
}

function ColorSchemeSetter(props: { darkMode: boolean }) {
  const colorScheme = useMantineColorScheme();
  // Update data attribute for color scheme.
  useEffect(() => {
    colorScheme.setColorScheme(props.darkMode ? "dark" : "light");
  }, [props.darkMode]);
  return null;
}

/**
 * Notifications panel with fixed styling.
 */
function NotificationsPanel() {
  return (
    <Notifications
      position="top-left"
      limit={10}
      containerWidth="20em"
      withinPortal={false}
      styles={{
        root: {
          boxShadow: "0.1em 0 1em 0 rgba(0,0,0,0.1) !important",
          position: "absolute",
          top: "1em",
          left: "1em",
          pointerEvents: "none",
        },
        notification: {
          pointerEvents: "all",
        },
      }}
    />
  );
}

/**
 * Main 3D canvas component.
 */
function ViewerCanvas({ children }: { children: React.ReactNode }) {
  const viewer = React.useContext(ViewerContext)!;
  const sendClickThrottled = useThrottledMessageSender(20);
  const theme = useMantineTheme();
  const { ref: inViewRef, inView } = useInView();

  // Memoize camera controls to prevent unnecessary re-creation.
  const memoizedCameraControls = useMemo(
    () => <SynchronizedCameraControls />,
    [],
  );

  // Handle pointer down event. I don't think we need useCallback here, since
  // remounts should be very rare.
  const handlePointerDown = (e: React.PointerEvent) => {
    const { mutable } = viewer;
    const pointerInfo = mutable.current.scenePointerInfo;
    if (pointerInfo.enabled === false) return;

    const canvasBbox = mutable.current.canvas!.getBoundingClientRect();
    pointerInfo.dragStart = [
      e.clientX - canvasBbox.left,
      e.clientY - canvasBbox.top,
    ];
    pointerInfo.dragEnd = pointerInfo.dragStart;

    if (ndcFromPointerXy(viewer, pointerInfo.dragEnd) === null) return;
    if (pointerInfo.isDragging) return;

    pointerInfo.isDragging = true;
    mutable.current.cameraControl!.enabled = false;

    const ctx = mutable.current.canvas2d!.getContext("2d")!;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  };

  // Handle pointer move event.
  const handlePointerMove = (e: React.PointerEvent) => {
    const { mutable } = viewer;
    const pointerInfo = mutable.current.scenePointerInfo;
    if (pointerInfo.enabled === false || !pointerInfo.isDragging) return;

    const canvasBbox = mutable.current.canvas!.getBoundingClientRect();
    const pointerXy: [number, number] = [
      e.clientX - canvasBbox.left,
      e.clientY - canvasBbox.top,
    ];

    if (ndcFromPointerXy(viewer, pointerXy) === null) return;
    pointerInfo.dragEnd = pointerXy;

    // Check if pointer moved enough to be considered a drag.
    if (
      Math.abs(pointerInfo.dragEnd[0] - pointerInfo.dragStart[0]) <= 3 &&
      Math.abs(pointerInfo.dragEnd[1] - pointerInfo.dragStart[1]) <= 3
    )
      return;

    // Draw selection rectangle if in rect-select mode.
    if (pointerInfo.enabled === "rect-select") {
      const ctx = mutable.current.canvas2d!.getContext("2d")!;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.beginPath();
      ctx.fillStyle = theme.primaryColor;
      ctx.strokeStyle = "blue";
      ctx.globalAlpha = 0.2;
      ctx.fillRect(
        pointerInfo.dragStart[0],
        pointerInfo.dragStart[1],
        pointerInfo.dragEnd[0] - pointerInfo.dragStart[0],
        pointerInfo.dragEnd[1] - pointerInfo.dragStart[1],
      );
      ctx.globalAlpha = 1.0;
      ctx.stroke();
    }
  };

  // Handle pointer up event.
  const handlePointerUp = () => {
    const { mutable } = viewer;
    const pointerInfo = mutable.current.scenePointerInfo;

    // Re-enable camera controls.
    mutable.current.cameraControl!.enabled = true;
    if (pointerInfo.enabled === false || !pointerInfo.isDragging) return;

    const ctx = mutable.current.canvas2d!.getContext("2d")!;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Handle click or rect-select based on mode.
    if (pointerInfo.enabled === "click") {
      sendClickMessage(viewer, pointerInfo.dragEnd, sendClickThrottled);
    } else if (pointerInfo.enabled === "rect-select") {
      sendRectSelectMessage(viewer, pointerInfo, sendClickThrottled);
    }

    pointerInfo.isDragging = false;
  };

  return (
    <div
      ref={inViewRef}
      style={{ position: "relative", zIndex: 0, width: "100%", height: "100%" }}
    >
      <Canvas
        camera={{ position: [-3.0, 3.0, -3.0], near: 0.01, far: 1000.0 }}
        gl={{ preserveDrawingBuffer: true }}
        style={{ width: "100%", height: "100%" }}
        ref={(el) => (viewer.mutable.current.canvas = el)}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        shadows
      >
        <Bvh firstHitOnly>
          {!inView && <DisableRender />}
          <BackgroundImage />
          <SceneContextSetter />
          {memoizedCameraControls}
          <SplatRenderContext>
            <AdaptiveDpr />
            {children}
            <SceneNodeThreeObject name="" />
          </SplatRenderContext>
          <DefaultLights />
        </Bvh>
      </Canvas>
    </div>
  );
}

// ======= Helper functions for pointer events. =======

/**
 * Send a click message based on the pointer position.
 */
function sendClickMessage(
  viewer: ViewerContextContents,
  pointerPos: [number, number],
  sendClickThrottled: (message: any) => void,
) {
  const raycaster = new THREE.Raycaster();
  const mouseVector = ndcFromPointerXy(viewer, pointerPos);
  if (mouseVector === null) return;

  raycaster.setFromCamera(mouseVector, viewer.mutable.current.camera!);
  const ray = rayToViserCoords(viewer, raycaster.ray);
  const mouseVectorOpenCV = opencvXyFromPointerXy(viewer, pointerPos);

  sendClickThrottled({
    type: "ScenePointerMessage",
    event_type: "click",
    ray_origin: [ray.origin.x, ray.origin.y, ray.origin.z],
    ray_direction: [ray.direction.x, ray.direction.y, ray.direction.z],
    screen_pos: [[mouseVectorOpenCV.x, mouseVectorOpenCV.y]],
  });
}

/**
 * Send a rectangle selection message based on drag start/end positions.
 */
function sendRectSelectMessage(
  viewer: ViewerContextContents,
  pointerInfo: { dragStart: [number, number]; dragEnd: [number, number] },
  sendClickThrottled: (message: any) => void,
) {
  const firstMouseVector = opencvXyFromPointerXy(viewer, pointerInfo.dragStart);
  const lastMouseVector = opencvXyFromPointerXy(viewer, pointerInfo.dragEnd);

  const x_min = Math.min(firstMouseVector.x, lastMouseVector.x);
  const x_max = Math.max(firstMouseVector.x, lastMouseVector.x);
  const y_min = Math.min(firstMouseVector.y, lastMouseVector.y);
  const y_max = Math.max(firstMouseVector.y, lastMouseVector.y);

  sendClickThrottled({
    type: "ScenePointerMessage",
    event_type: "rect-select",
    ray_origin: null,
    ray_direction: null,
    screen_pos: [
      [x_min, y_min],
      [x_max, y_max],
    ],
  });
}

/**
 * DefaultLights component - handles environment map and lights.
 */
function DefaultLights() {
  const viewer = React.useContext(ViewerContext)!;
  const enableDefaultLights = viewer.useSceneTree(
    (state) => state.enableDefaultLights,
  );
  const enableDefaultLightsShadows = viewer.useSceneTree(
    (state) => state.enableDefaultLightsShadows,
  );
  const environmentMap = viewer.useSceneTree((state) => state.environmentMap);

  // Track environment rotation state.
  const [worldRotation, setWorldRotation] = useState(
    viewer.mutable.current.nodeAttributesFromName[""]!.wxyz!,
  );

  // Update rotation when changed.
  useFrame(() => {
    const currentRotation =
      viewer.mutable.current.nodeAttributesFromName[""]!.wxyz!;
    if (currentRotation !== worldRotation) {
      setWorldRotation(currentRotation);
    }
  });

  // Calculate environment map.
  const envMapNode = useMemo(() => {
    if (environmentMap.hdri === null) return null;

    // HDRI presets mapping.
    const presetsObj = {
      apartment: "lebombo_1k.hdr",
      city: "potsdamer_platz_1k.hdr",
      dawn: "kiara_1_dawn_1k.hdr",
      forest: "forest_slope_1k.hdr",
      lobby: "st_fagans_interior_1k.hdr",
      night: "dikhololo_night_1k.hdr",
      park: "rooitou_park_1k.hdr",
      studio: "studio_small_03_1k.hdr",
      sunset: "venice_sunset_1k.hdr",
      warehouse: "empty_warehouse_01_1k.hdr",
    };

    // Calculate quaternions for world transformation.
    const Rquat_threeworld_world = new THREE.Quaternion(
      worldRotation[1],
      worldRotation[2],
      worldRotation[3],
      worldRotation[0],
    );
    const Rquat_world_threeworld = Rquat_threeworld_world.clone().invert();

    // Calculate background rotation.
    const backgroundRotation = new THREE.Euler().setFromQuaternion(
      new THREE.Quaternion(
        environmentMap.background_wxyz[1],
        environmentMap.background_wxyz[2],
        environmentMap.background_wxyz[3],
        environmentMap.background_wxyz[0],
      )
        .premultiply(Rquat_threeworld_world)
        .multiply(Rquat_world_threeworld),
    );

    // Calculate environment rotation.
    const environmentRotation = new THREE.Euler().setFromQuaternion(
      new THREE.Quaternion(
        environmentMap.environment_wxyz[1],
        environmentMap.environment_wxyz[2],
        environmentMap.environment_wxyz[3],
        environmentMap.environment_wxyz[0],
      )
        .premultiply(Rquat_threeworld_world)
        .multiply(Rquat_world_threeworld),
    );

    return (
      <Environment
        files={`hdri/${presetsObj[environmentMap.hdri]}`}
        background={environmentMap.background}
        backgroundBlurriness={environmentMap.background_blurriness}
        backgroundIntensity={environmentMap.background_intensity}
        backgroundRotation={backgroundRotation}
        environmentIntensity={environmentMap.environment_intensity}
        environmentRotation={environmentRotation}
      />
    );
  }, [environmentMap, worldRotation]);

  // Return environment map only if lights are disabled.
  if (!enableDefaultLights) return envMapNode;

  // Return lights and environment map.
  return (
    <>
      <CsmDirectionalLight
        fade={true}
        lightIntensity={3.0}
        position={[-0.2, 1.0, -0.2]}
        cascades={3}
        color={0xffffff}
        maxFar={20}
        mode="practical"
        shadowBias={-0.0001}
        castShadow={enableDefaultLightsShadows}
      />
      <CsmDirectionalLight
        color={0xffffff}
        lightIntensity={0.4}
        position={[0, -1, 0]}
        castShadow={false}
      />
      {envMapNode}
    </>
  );
}

/**
 * Adaptive DPR component for performance optimization.
 */
function AdaptiveDpr() {
  const setDpr = useThree((state) => state.setDpr);

  return (
    <PerformanceMonitor
      factor={1.0}
      step={0.2}
      bounds={(refreshrate) => {
        const max = Math.min(refreshrate * 0.75, 85);
        const min = Math.max(max * 0.5, 38);
        return [min, max];
      }}
      onChange={({ factor, fps, refreshrate }) => {
        const dpr = window.devicePixelRatio * (0.2 + 0.8 * factor);
        console.log(
          `[Performance] Setting DPR to ${dpr}; FPS=${fps}/${refreshrate}`,
        );
        setDpr(dpr);
      }}
    />
  );
}

/**
 * 2D canvas overlay for drawing selection rectangles.
 */
function Viewer2DCanvas() {
  const viewer = React.useContext(ViewerContext)!;

  useEffect(() => {
    const canvas = viewer.mutable.current.canvas2d!;

    // Create a resize observer to update canvas dimensions.
    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      canvas.width = width;
      canvas.height = height;
    });

    resizeObserver.observe(canvas);
    return () => resizeObserver.disconnect();
  }, []);

  return (
    <canvas
      ref={(el) => (viewer.mutable.current.canvas2d = el)}
      style={{
        position: "absolute",
        zIndex: 1,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}

/**
 * Background image component with depth support.
 */
function BackgroundImage() {
  // Shader for background image with depth.
  const shaders = useMemo(
    () => ({
      vert: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
    `,
      frag: `
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
      float depth = rgbPacked.r * 0.00255 + rgbPacked.g * 0.6528 + rgbPacked.b * 167.1168;
      return depth;
    }

    void main() {
      if (!enabled) {
        discard;
      }
      vec4 color = texture(colorMap, vUv);
      gl_FragColor = vec4(color.rgb, 1.0);

      float bufDepth;
      if(hasDepth){
        float depth = readDepth(depthMap, vUv);
        bufDepth = viewZToPerspectiveDepth(-depth, cameraNear, cameraFar);
      } else {
        bufDepth = 1.0;
      }
      gl_FragDepth = bufDepth;
    }
    `,
    }),
    [],
  );

  // Create material.
  const backgroundMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        fragmentShader: shaders.frag,
        vertexShader: shaders.vert,
        uniforms: {
          enabled: { value: false },
          depthMap: { value: null },
          colorMap: { value: null },
          cameraNear: { value: null },
          cameraFar: { value: null },
          hasDepth: { value: false },
        },
      }),
    [shaders],
  );

  // Store material in viewer context.
  const { mutable } = React.useContext(ViewerContext)!;
  mutable.current.backgroundMaterial = backgroundMaterial;
  const backgroundMesh = React.useRef<THREE.Mesh>(null);

  // Update position and rotation in render loop.
  useFrame(({ camera }) => {
    if (!(camera instanceof THREE.PerspectiveCamera)) {
      console.error(
        "Camera is not a perspective camera, cannot render background image.",
      );
      return;
    }

    const mesh = backgroundMesh.current!;

    // Position behind camera.
    const lookdir = camera.getWorldDirection(new THREE.Vector3());
    mesh.position.copy(camera.position).addScaledVector(lookdir, 1.0);
    mesh.quaternion.copy(camera.quaternion);

    // Size based on camera parameters.
    const f = camera.getFocalLength();
    mesh.scale.set(camera.getFilmWidth() / f, camera.getFilmHeight() / f, 1.0);

    // Update shader uniforms.
    backgroundMaterial.uniforms.cameraNear.value = camera.near;
    backgroundMaterial.uniforms.cameraFar.value = camera.far;
  });

  return (
    <mesh ref={backgroundMesh} material={backgroundMaterial}>
      <planeGeometry attach="geometry" args={[1, 1]} />
    </mesh>
  );
}

/**
 * Helper component to sync scene and camera state.
 */
function SceneContextSetter() {
  const { mutable } = React.useContext(ViewerContext)!;
  mutable.current.scene = useThree((state) => state.scene);
  mutable.current.camera = useThree(
    (state) => state.camera as THREE.PerspectiveCamera,
  );
  return null;
}

/**
 * Viser logo with about modal.
 */
function ViserLogo() {
  const [aboutModalOpened, { open: openAbout, close: closeAbout }] =
    useDisclosure(false);

  return (
    <>
      <Tooltip label={`Viser ${VISER_VERSION}`}>
        <Box
          style={{
            position: "absolute",
            bottom: "1em",
            left: "1em",
            cursor: "pointer",
          }}
          component="a"
          onClick={openAbout}
          title="About Viser"
        >
          <Image src="./logo.svg" style={{ width: "2.5em", height: "auto" }} />
        </Box>
      </Tooltip>
      <Modal
        opened={aboutModalOpened}
        onClose={closeAbout}
        withCloseButton={false}
        size="xl"
        style={{ textAlign: "center" }}
      >
        <Box>
          <p>Viser is a 3D visualization toolkit developed at UC Berkeley.</p>
          <p>
            <Anchor
              href="https://github.com/nerfstudio-project/"
              target="_blank"
              style={{ fontWeight: "600" }}
            >
              Nerfstudio
            </Anchor>
            &nbsp;&nbsp;&bull;&nbsp;&nbsp;
            <Anchor
              href="https://github.com/nerfstudio-project/viser"
              target="_blank"
              style={{ fontWeight: "600" }}
            >
              GitHub
            </Anchor>
            &nbsp;&nbsp;&bull;&nbsp;&nbsp;
            <Anchor
              href="https://viser.studio/main"
              target="_blank"
              style={{ fontWeight: "600" }}
            >
              Documentation
            </Anchor>
          </p>
        </Box>
      </Modal>
    </>
  );
}
