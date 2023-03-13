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
import { PerspectiveCamera } from "three";
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

  function sendCamera() {
    if (rootState.current === null) return;
    const three_camera = rootState.current.camera as PerspectiveCamera;
    const message: ViewerCameraMessage = {
      type: "viewer_camera",
      wxyz: [
        three_camera.quaternion.w,
        three_camera.quaternion.x,
        three_camera.quaternion.y,
        three_camera.quaternion.z,
      ],
      position: three_camera.position.toArray(),
      aspect: three_camera.aspect,
      fov: (three_camera.fov * Math.PI) / 180.0,
    };
    const websocket = props.websocketRef.current;
    websocket && websocket.send(encode(message));
  }

  return (
    <OrbitControls
      minDistance={0.5}
      maxDistance={200.0}
      enableDamping={true}
      dampingFactor={0.2}
      onChange={() => {
        if (cameraThrottleReady.current) {
          sendCamera();
          cameraThrottleReady.current = false;
          cameraThrottleStale.current = false;
          setTimeout(() => {
            cameraThrottleReady.current = true;
            if (cameraThrottleStale.current) sendCamera();
          }, 50);
        } else {
          cameraThrottleStale.current = true;
        }
      }}
    />
  );
}

function Root() {
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

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
