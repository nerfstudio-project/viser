import styled from "@emotion/styled";
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree, RootState } from "@react-three/fiber";
import React, { MutableRefObject } from "react";
import { createRoot } from "react-dom/client";

import ControlPanel from "./ControlPanel";
import LabelRenderer from "./LabelRenderer";
import { SceneNodeThreeObject, useSceneTreeState } from "./SceneTree";

import "./index.css";

import Box from "@mui/material/Box";
import { Euler, PerspectiveCamera, Quaternion } from "three";
import { ViewerCameraMessage } from "./WebsocketMessages";
import { encode } from "@msgpack/msgpack";

interface CameraSynchronizerProps {
  websocketRef: MutableRefObject<WebSocket | null>;
}

// Communicate the threejs camera to the server.
//
// We may want to add the ability to make this opt-in.
function CameraSynchronizer(props: CameraSynchronizerProps) {
  const rootState = React.useRef<RootState | null>(null);
  useThree((state) => {
    rootState.current = state;
  });

  const cameraThrottleReady = React.useRef(true);
  const cameraThrottleStale = React.useRef(false);

  // We put Z up to match the scene tree, and convert threejs camera convention
  // to the OpenCV one.
  const R_threecam_cam = new Quaternion();
  const R_worldfix_world = new Quaternion();
  R_threecam_cam.setFromEuler(new Euler(Math.PI, 0.0, 0.0));
  R_worldfix_world.setFromEuler(new Euler(Math.PI / 2.0, 0.0, 0.0));

  function sendCamera() {
    if (rootState.current === null) return;
    const three_camera = rootState.current.camera as PerspectiveCamera;

    const R_world_camera = R_worldfix_world.clone()
      .multiply(three_camera.quaternion)
      .multiply(R_threecam_cam);

    const message: ViewerCameraMessage = {
      type: "viewer_camera",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: three_camera.position
        .clone()
        .applyQuaternion(R_worldfix_world)
        .toArray(),
      aspect: three_camera.aspect,
      fov: (three_camera.fov * Math.PI) / 180.0,
    };
    const websocket = props.websocketRef.current;
    websocket && websocket.send(encode(message));
  }

  // Send camera for new connections. Slightly hacky!
  React.useEffect(() => {
    let disconnected_prev = props.websocketRef.current === null;
    const interval = setInterval(() => {
      let disconnected = props.websocketRef.current === null;
      if (!disconnected && disconnected_prev) {
        sendCamera();
      }
      disconnected_prev = disconnected;
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  });

  return (
    <OrbitControls
      minDistance={0.5}
      maxDistance={200.0}
      enableDamping={false}
      onChange={() => {
        if (cameraThrottleReady.current) {
          sendCamera();
          cameraThrottleReady.current = false;
          cameraThrottleStale.current = false;
          setTimeout(() => {
            cameraThrottleReady.current = true;
            if (cameraThrottleStale.current) sendCamera();
          }, 10);
        } else {
          cameraThrottleStale.current = true;
        }
      }}
    />
  );
}

function SingleViewer() {
  // Layout and styles.
  const Wrapper = styled(Box)`
    width: 100%;
    height: 100%;
    position: relative;
  `;

  const Viewport = styled(Canvas)`
    position: relative;
    z-index: 0;

    width: 100%;
    height: 100%;
  `;

  // Our 2D label renderer needs access to the div used for rendering.
  const wrapperRef = React.useRef<HTMLDivElement>(null);
  const websocketRef = React.useRef<WebSocket | null>(null);

  // Declare the scene tree state. This returns a zustand store/hook, which we
  // can pass to any children that need state access.
  const useSceneTree = useSceneTreeState();

  // <Stats showPanel={0} className="stats" />
  // <gridHelper args={[10.0, 10]} />
  return (
    <Wrapper ref={wrapperRef}>
      <ControlPanel
        wrapperRef={wrapperRef}
        websocketRef={websocketRef}
        useSceneTree={useSceneTree}
      />
      <Viewport>
        <LabelRenderer wrapperRef={wrapperRef} />
        <CameraSynchronizer websocketRef={websocketRef} />
        <SceneNodeThreeObject id={0} useSceneTree={useSceneTree} />
      </Viewport>
    </Wrapper>
  );
}

function Root() {
  return (
    <>
      <Box
        component="div"
        sx={{ position: "absolute", left: 0, height: "100%", width: "50%" }}
      >
        <SingleViewer />
      </Box>
      <Box
        component="div"
        sx={{ position: "absolute", right: 0, height: "100%", width: "50%" }}
      >
        <SingleViewer />
      </Box>
    </>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
