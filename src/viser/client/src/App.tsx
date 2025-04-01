// @refresh reset
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./App.css";
import { useInView } from "react-intersection-observer";

import { Notifications } from "@mantine/notifications";

import { Environment, PerformanceMonitor, Stats, Bvh } from "@react-three/drei";
import * as THREE from "three";
import { Canvas, useThree, useFrame } from "@react-three/fiber";

import { SynchronizedCameraControls } from "./CameraControls";
import {
  Anchor,
  Box,
  ColorSchemeScript,
  Image,
  MantineProvider,
  Modal,
  Tooltip,
  createTheme,
  useMantineTheme,
} from "@mantine/core";
import React, { useEffect } from "react";
import { SceneNodeThreeObject } from "./SceneTree";
import { ViewerContext, ViewerContextContents } from "./ViewerContext";

import "./index.css";

import ControlPanel from "./ControlPanel/ControlPanel";
import { useGuiState } from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";
import { WebsocketMessageProducer } from "./WebsocketInterface";
import { Titlebar } from "./Titlebar";
import { ViserModal } from "./Modal";
import { useSceneTreeState } from "./SceneTreeState";
import { useThrottledMessageSender } from "./WebsocketFunctions";
import { useDisclosure } from "@mantine/hooks";
import { rayToViserCoords } from "./WorldTransformUtils";
import { ndcFromPointerXy, opencvXyFromPointerXy } from "./ClickUtils";
import { theme } from "./AppTheme";
import { FrameSynchronizedMessageHandler } from "./MessageHandler";
import { PlaybackFromFile } from "./FilePlayback";
import { SplatRenderContext } from "./Splatting/GaussianSplats";
import { BrowserWarning } from "./BrowserWarning";
import { MacWindowWrapper } from "./MacWindowWrapper";
import { CsmDirectionalLight } from "./CsmDirectionalLight";

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

  // Playback mode for embedding viser.
  const searchParams = new URLSearchParams(window.location.search);
  const playbackPath = searchParams.get("playbackPath");
  const darkMode = searchParams.get("darkMode") !== null;
  const showStats = searchParams.get("showStats") !== null;

  // Values that can be globally accessed by components in a viewer.
  const nodeRefFromName = React.useRef<{
    [name: string]: undefined | THREE.Object3D;
  }>({});
  const viewer: ViewerContextContents = {
    messageSource: playbackPath === null ? "websocket" : "file_playback",
    useSceneTree: useSceneTreeState(nodeRefFromName),
    useGui: useGuiState(initialServer),
    sendMessageRef: React.useRef(
      playbackPath == null
        ? (message) =>
            console.log(
              `Tried to send ${message.type} but websocket is not connected!`,
            )
        : () => null,
    ),
    canvasRef: React.useRef(null),
    sceneRef: React.useRef(null),
    cameraRef: React.useRef(null),
    backgroundMaterialRef: React.useRef(null),
    cameraControlRef: React.useRef(null),
    sendCameraRef: React.useRef(null),
    resetCameraViewRef: React.useRef(null),
    // Scene node attributes that aren't placed in the zustand state for performance reasons.
    nodeAttributesFromName: React.useRef({
      "": {
        wxyz: (() => {
          const quat = new THREE.Quaternion().setFromEuler(
            new THREE.Euler(Math.PI / 2, Math.PI, -Math.PI / 2),
          );
          return [quat.w, quat.x, quat.y, quat.z];
        })(),
      },
    }),
    nodeRefFromName: nodeRefFromName,
    messageQueueRef: React.useRef([]),
    getRenderRequestState: React.useRef("ready"),
    getRenderRequest: React.useRef(null),
    scenePointerInfo: React.useRef({
      enabled: false,
      dragStart: [0, 0],
      dragEnd: [0, 0],
      isDragging: false,
    }),
    canvas2dRef: React.useRef(null),
    skinnedMeshState: React.useRef({}),
    // Global hover state tracking for cursor management
    hoveredElementsCount: React.useRef(0),
  };

  // Set dark default if specified in URL.
  if (darkMode) viewer.useGui.getState().theme.dark_mode = darkMode;

  return (
    <ViewerContext.Provider value={viewer}>
      <ViewerContents>
        {viewer.messageSource === "websocket" ? (
          <WebsocketMessageProducer />
        ) : null}
        {viewer.messageSource === "file_playback" ? (
          <PlaybackFromFile fileUrl={playbackPath!} />
        ) : null}
        {showStats ? <Stats className="stats-panel" /> : null}
      </ViewerContents>
    </ViewerContext.Provider>
  );
}

function ViewerContents({ children }: { children: React.ReactNode }) {
  const viewer = React.useContext(ViewerContext)!;
  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const colors = viewer.useGui((state) => state.theme.colors);
  const controlLayout = viewer.useGui((state) => state.theme.control_layout);
  return (
    <>
      <ColorSchemeScript forceColorScheme={darkMode ? "dark" : "light"} />
      <MantineProvider
        theme={createTheme({
          ...theme,
          ...(colors === null
            ? {}
            : { colors: { custom: colors }, primaryColor: "custom" }),
        })}
        forceColorScheme={darkMode ? "dark" : "light"}
      >
        {children}
        <Notifications
          position="top-left"
          limit={10}
          containerWidth="20em"
          styles={{
            root: {
              boxShadow: "0.1em 0 1em 0 rgba(0,0,0,0.1) !important",
            },
          }}
        />
        <BrowserWarning />
        <ViserModal />
        <Box
          style={{
            width: "100%",
            height: "100%",
            // We use flex display for the titlebar layout.
            display: "flex",
            position: "relative",
            flexDirection: "column",
          }}
        >
          <Titlebar />
          <Box
            style={{
              // Put the canvas and control panel side-by-side.
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
              {viewer.useGui((state) => state.theme.show_logo) &&
              viewer.messageSource == "websocket" ? (
                <ViserLogo />
              ) : null}
            </Box>
            {viewer.messageSource == "websocket" ? (
              <ControlPanel control_layout={controlLayout} />
            ) : null}
          </Box>
        </Box>
      </MantineProvider>
    </>
  );
}

const DisableRender = () => useFrame(() => null, 1000);

function ViewerCanvas({ children }: { children: React.ReactNode }) {
  const viewer = React.useContext(ViewerContext)!;
  const sendClickThrottled = useThrottledMessageSender(20);
  const theme = useMantineTheme();

  // Make sure we don't re-mount the camera controls, since that will reset the camera position.
  const memoizedCameraControls = React.useMemo(
    () => <SynchronizedCameraControls />,
    [],
  );

  // We'll disable rendering if the canvas is not in view.
  const { ref: inViewRef, inView } = useInView();

  return (
    <div
      ref={inViewRef}
      style={{
        position: "relative",
        zIndex: 0,
        width: "100%",
        height: "100%",
      }}
    >
      <Canvas
        camera={{ position: [-3.0, 3.0, -3.0], near: 0.01, far: 1000.0 }}
        gl={{ preserveDrawingBuffer: true }}
        style={{
          width: "100%",
          height: "100%",
        }}
        ref={viewer.canvasRef}
        // Handle scene click events (onPointerDown, onPointerMove, onPointerUp)
        onPointerDown={(e) => {
          const pointerInfo = viewer.scenePointerInfo.current!;

          // Only handle pointer events if enabled.
          if (pointerInfo.enabled === false) return;

          // Keep track of the first click position.
          const canvasBbox = viewer.canvasRef.current!.getBoundingClientRect();
          pointerInfo.dragStart = [
            e.clientX - canvasBbox.left,
            e.clientY - canvasBbox.top,
          ];
          pointerInfo.dragEnd = pointerInfo.dragStart;

          // Check if pointer position is in bounds.
          if (ndcFromPointerXy(viewer, pointerInfo.dragEnd) === null) return;

          // Only allow one drag event at a time.
          if (pointerInfo.isDragging) return;
          pointerInfo.isDragging = true;

          // Disable camera controls -- we don't want the camera to move while we're dragging.
          viewer.cameraControlRef.current!.enabled = false;

          const ctx = viewer.canvas2dRef.current!.getContext("2d")!;
          ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        }}
        onPointerMove={(e) => {
          const pointerInfo = viewer.scenePointerInfo.current!;

          // Only handle if click events are enabled, and if pointer is down (i.e., dragging).
          if (pointerInfo.enabled === false || !pointerInfo.isDragging) return;

          // Check if pointer position is in boudns.
          const canvasBbox = viewer.canvasRef.current!.getBoundingClientRect();
          const pointerXy: [number, number] = [
            e.clientX - canvasBbox.left,
            e.clientY - canvasBbox.top,
          ];
          if (ndcFromPointerXy(viewer, pointerXy) === null) return;

          // Check if mouse position has changed sufficiently from last position.
          // Uses 3px as a threshood, similar to drag detection in
          // `SceneNodeClickMessage` from `SceneTree.tsx`.
          pointerInfo.dragEnd = pointerXy;
          if (
            Math.abs(pointerInfo.dragEnd[0] - pointerInfo.dragStart[0]) <= 3 &&
            Math.abs(pointerInfo.dragEnd[1] - pointerInfo.dragStart[1]) <= 3
          )
            return;

          // If we're listening for scene box events, draw the box on the 2D canvas for user feedback.
          if (pointerInfo.enabled === "rect-select") {
            const ctx = viewer.canvas2dRef.current!.getContext("2d")!;
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
        }}
        onPointerUp={() => {
          const pointerInfo = viewer.scenePointerInfo.current!;

          // Re-enable camera controls! Was disabled in `onPointerDown`, to allow
          // for mouse drag w/o camera movement.
          viewer.cameraControlRef.current!.enabled = true;

          // Only handle if click events are enabled, and if pointer was down (i.e., dragging).
          if (pointerInfo.enabled === false || !pointerInfo.isDragging) return;

          const ctx = viewer.canvas2dRef.current!.getContext("2d")!;
          ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

          // If there's only one pointer, send a click message.
          // The message will return origin/direction lists of length 1.
          if (pointerInfo.enabled === "click") {
            const raycaster = new THREE.Raycaster();

            // Raycaster expects NDC coordinates, so we convert the click event to NDC.
            const mouseVector = ndcFromPointerXy(viewer, pointerInfo.dragEnd);
            if (mouseVector === null) return;
            raycaster.setFromCamera(mouseVector, viewer.cameraRef.current!);
            const ray = rayToViserCoords(viewer, raycaster.ray);

            // Send OpenCV image coordinates to the server (normalized).
            const mouseVectorOpenCV = opencvXyFromPointerXy(
              viewer,
              pointerInfo.dragEnd,
            );

            sendClickThrottled({
              type: "ScenePointerMessage",
              event_type: "click",
              ray_origin: [ray.origin.x, ray.origin.y, ray.origin.z],
              ray_direction: [
                ray.direction.x,
                ray.direction.y,
                ray.direction.z,
              ],
              screen_pos: [[mouseVectorOpenCV.x, mouseVectorOpenCV.y]],
            });
          } else if (pointerInfo.enabled === "rect-select") {
            // If the ScenePointerEvent had mouse drag movement, we will send a "box" message:
            // Use the first and last mouse positions to create a box.
            // Again, click should be in openCV image coordinates (normalized).
            const firstMouseVector = opencvXyFromPointerXy(
              viewer,
              pointerInfo.dragStart,
            );
            const lastMouseVector = opencvXyFromPointerXy(
              viewer,
              pointerInfo.dragEnd,
            );

            const x_min = Math.min(firstMouseVector.x, lastMouseVector.x);
            const x_max = Math.max(firstMouseVector.x, lastMouseVector.x);
            const y_min = Math.min(firstMouseVector.y, lastMouseVector.y);
            const y_max = Math.max(firstMouseVector.y, lastMouseVector.y);

            // Send the upper-left and lower-right corners of the box.
            const screenBoxList: [number, number][] = [
              [x_min, y_min],
              [x_max, y_max],
            ];

            sendClickThrottled({
              type: "ScenePointerMessage",
              event_type: "rect-select",
              ray_origin: null,
              ray_direction: null,
              screen_pos: screenBoxList,
            });
          }

          // Release drag lock.
          pointerInfo.isDragging = false;
        }}
        shadows
      >
        <Bvh firstHitOnly>
          {inView ? null : <DisableRender />}
          <BackgroundImage />
          <SceneContextSetter />
          {memoizedCameraControls}
          <SplatRenderContext>
            <AdaptiveDpr />
            {children}
            <SceneNodeThreeObject name="" parent={null} />
          </SplatRenderContext>
          <DefaultLights />
        </Bvh>
      </Canvas>
    </div>
  );
}

function DefaultLights() {
  const viewer = React.useContext(ViewerContext)!;
  const enableDefaultLights = viewer.useSceneTree(
    (state) => state.enableDefaultLights,
  );
  const enableDefaultLightsShadows = viewer.useSceneTree(
    (state) => state.enableDefaultLightsShadows,
  );
  const environmentMap = viewer.useSceneTree((state) => state.environmentMap);

  // Environment map frames:
  // - We want the `background_wxyz` and `environment_wxyz` to be in the Viser
  //   world frame. This is different from the threejs world frame, which should
  //   not be exposed to the user.
  // - `backgroundRotation` and `environmentRotation` for the `Environment` component
  //   are in the threejs world frame.
  const [R_threeworld_world, setR_threeworld_world] = React.useState(
    // In Python, this will be set by `set_up_direction()`.
    viewer.nodeAttributesFromName.current![""]!.wxyz!,
  );
  useFrame(() => {
    const currentR_threeworld_world =
      viewer.nodeAttributesFromName.current![""]!.wxyz!;
    if (currentR_threeworld_world !== R_threeworld_world) {
      setR_threeworld_world(currentR_threeworld_world);
    }
  });

  const Rquat_threeworld_world = new THREE.Quaternion(
    R_threeworld_world[1],
    R_threeworld_world[2],
    R_threeworld_world[3],
    R_threeworld_world[0],
  );
  const Rquat_world_threeworld = Rquat_threeworld_world.clone().invert();
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

  let envMapNode;
  if (environmentMap.hdri === null) {
    envMapNode = null;
  } else {
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
    envMapNode = (
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
  }
  // TODO: need to figure out lights
  if (enableDefaultLights)
    return (
      <>
        <CsmDirectionalLight
          fade={true}
          lightIntensity={3.0}
          position={[-0.2, 1.0, -0.2]} // Coming from above, slightly off-center
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
          position={[0, -1, 0]} // Light from below
          castShadow={false /* Let's only cast a shadow from above. */}
        />
        {envMapNode}
      </>
    );
  else return envMapNode;
}

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

/* HTML Canvas, for drawing 2D. */
function Viewer2DCanvas() {
  const viewer = React.useContext(ViewerContext)!;
  useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      canvas.width = width;
      canvas.height = height;
    });

    // Observe the canvas.
    const canvas = viewer.canvas2dRef.current!;
    resizeObserver.observe(canvas);

    // Cleanup
    return () => resizeObserver.disconnect();
  });
  return (
    <canvas
      ref={viewer.canvas2dRef}
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
    <mesh ref={backgroundMesh} material={backgroundMaterial}>
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
  // Parse dummy window dimensions from URL if present
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

  // If dummy window dimensions are specified, wrap content in MacWindowWrapper
  if (dummyWindowParam) {
    const [width, height] = dummyWindowParam.split("x").map(Number);
    if (!isNaN(width) && !isNaN(height)) {
      return (
        <MacWindowWrapper
          title={dummyWindowTitle}
          width={width}
          height={height}
        >
          {content}
        </MacWindowWrapper>
      );
    }
  }
  return content;
}

/** Logo. When clicked, opens an info modal. */
function ViserLogo() {
  const [aboutModalOpened, { open: openAbout, close: closeAbout }] =
    useDisclosure(false);
  return (
    <>
      <Tooltip label="About Viser">
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
        ta="center"
      >
        <Box>
          <p>Viser is a 3D visualization toolkit developed at UC Berkeley.</p>
          <p>
            <Anchor
              href="https://github.com/nerfstudio-project/"
              target="_blank"
              fw="600"
              style={{ "&:focus": { outline: "none" } }}
            >
              Nerfstudio
            </Anchor>
            &nbsp;&nbsp;&bull;&nbsp;&nbsp;
            <Anchor
              href="https://github.com/nerfstudio-project/viser"
              target="_blank"
              fw="600"
              style={{ "&:focus": { outline: "none" } }}
            >
              GitHub
            </Anchor>
            &nbsp;&nbsp;&bull;&nbsp;&nbsp;
            <Anchor
              href="https://viser.studio/main"
              target="_blank"
              fw="600"
              style={{ "&:focus": { outline: "none" } }}
            >
              Documentation
            </Anchor>
          </p>
        </Box>
      </Modal>
    </>
  );
}
