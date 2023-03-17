import styled from "@emotion/styled";
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import React, { MutableRefObject, RefObject, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { OrbitControls as OrbitControls_ } from "three-stdlib";

import ControlPanel from "./ControlPanel";
import LabelRenderer from "./LabelRenderer";
import { SceneNodeThreeObject, useSceneTreeState } from "./SceneTree";

import "./index.css";

import Box from "@mui/material/Box";
import { Euler, PerspectiveCamera, Quaternion } from "three";
import { ViewerCameraMessage } from "./WebsocketMessages";
import { encode } from "@msgpack/msgpack";
import { IconButton, useMediaQuery } from "@mui/material";
import { RemoveCircleRounded, AddCircleRounded } from "@mui/icons-material";

interface SynchronizedOrbitControlsProps {
  globalCameras: MutableRefObject<CameraPrimitives[]>;
  websocketRef: MutableRefObject<WebSocket | null>;
}

// Communicate the threejs camera to the server.
function SynchronizedOrbitControls(props: SynchronizedOrbitControlsProps) {
  console.log("Setting up camera synchronizer; this should only happen once!");

  const camera = useThree((state) => state.camera as PerspectiveCamera);
  const cameraThrottleReady = React.useRef(true);
  const cameraThrottleStale = React.useRef(false);

  const orbitRef = React.useRef<OrbitControls_>(null);

  // We put Z up to match the scene tree, and convert threejs camera convention
  // to the OpenCV one.
  const R_threecam_cam = new Quaternion();
  const R_worldfix_world = new Quaternion();
  R_threecam_cam.setFromEuler(new Euler(Math.PI, 0.0, 0.0));
  R_worldfix_world.setFromEuler(new Euler(Math.PI / 2.0, 0.0, 0.0));

  function sendCamera() {
    const three_camera = camera;

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

  // What do we need to when the camera moves?
  function cameraChangedCallback() {
    // Match all cameras.
    props.globalCameras.current.forEach((other) => {
      if (camera === other.camera) return;
      other.camera.copy(camera);
      other.orbitRef.current!.target.copy(orbitRef.current!.target);
    });

    // If desired, send our camera via websocket.
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
  }

  // Send camera for new connections. Slightly hacky!
  React.useEffect(() => {
    const cameraMeta: CameraPrimitives = { camera: camera, orbitRef: orbitRef };

    if (props.globalCameras.current.length > 0) {
      const other = props.globalCameras.current[0];
      camera.copy(other.camera);
      orbitRef.current!.target.copy(other.orbitRef.current!.target);
    }

    props.globalCameras.current.push(cameraMeta);

    let disconnected_prev = props.websocketRef.current === null;
    const interval = setInterval(() => {
      let disconnected = props.websocketRef.current === null;
      if (!disconnected && disconnected_prev) {
        sendCamera();
      }
      disconnected_prev = disconnected;
    }, 1000);

    window.addEventListener("resize", cameraChangedCallback);

    return () => {
      window.removeEventListener("resize", cameraChangedCallback);

      clearInterval(interval);

      // Remove ourself from camera list. Since we always add/remove panels
      // from the end, a pop() would actually work as well here in constant
      // time.
      props.globalCameras.current.splice(
        props.globalCameras.current.indexOf(cameraMeta),
        1
      );
    };
  });
  return (
    <OrbitControls
      ref={orbitRef}
      minDistance={0.5}
      maxDistance={200.0}
      enableDamping={false}
      onChange={cameraChangedCallback}
    />
  );
}

interface SingleViewerProps {
  panelKey: number;
  globalCameras: MutableRefObject<CameraPrimitives[]>;
}

const SingleViewer = React.memo(function SingleViewer(
  props: SingleViewerProps
) {
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
        <SynchronizedOrbitControls
          websocketRef={websocketRef}
          globalCameras={props.globalCameras}
        />
        <SceneNodeThreeObject id={0} useSceneTree={useSceneTree} />
      </Viewport>
    </Wrapper>
  );
});

interface CameraPrimitives {
  camera: PerspectiveCamera;
  orbitRef: RefObject<OrbitControls_>;
}

function Root() {
  const globalCameras = useRef<CameraPrimitives[]>([]);
  const [panelCount, setPanelCount] = useState(1);
  const isPortrait = useMediaQuery("(orientation: portrait)");
  return (
    <Box
      component="div"
      sx={{ width: "100%", height: "100%", position: "relative" }}
    >
      <Box
        component="div"
        sx={{ position: "fixed", top: "1em", left: "1em", zIndex: "1000" }}
      >
        <IconButton>
          <AddCircleRounded
            onClick={() => {
              setPanelCount(panelCount + 1);
            }}
          />
        </IconButton>
        <IconButton disabled={panelCount == 1}>
          <RemoveCircleRounded
            onClick={() => {
              if (panelCount === 1) return;
              setPanelCount(panelCount - 1);
            }}
          />
        </IconButton>
      </Box>
      {Array.from({ length: panelCount }, (_, i) => {
        return (
          <Box
            component="div"
            key={"box-" + i.toString()}
            sx={
              isPortrait
                ? {
                    width: "100%",
                    height: (100.0 / panelCount).toString() + "%",
                  }
                : {
                    height: "100%",
                    float: "left",
                    width: (100.0 / panelCount).toString() + "%",
                  }
            }
          >
            <SingleViewer panelKey={i} globalCameras={globalCameras} />
          </Box>
        );
      })}
    </Box>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
