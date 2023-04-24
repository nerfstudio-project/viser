import React, { RefObject, useContext } from "react";
import { PerspectiveCamera } from "three";
import * as holdEvent from "hold-event";
import { CameraControls } from "@react-three/drei";
import { makeThrottledMessageSender } from "./WebsocketInterface";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { ViewerContext } from ".";

export interface CameraPrimitives {
  synchronize: boolean;
  cameras: PerspectiveCamera[];
  cameraControlRefs: RefObject<CameraControls>[];
}

/** OrbitControls, but synchronized with the server and other panels. */
export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);
  const cameraControlRef = React.useRef<CameraControls>(null);

  const sendCameraThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    20
  );

  // Callback for sending cameras.
  const sendCamera = React.useCallback(() => {
    const three_camera = camera;

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const R_threecam_cam = new THREE.Quaternion();
    const R_world_threeworld = new THREE.Quaternion();
    R_threecam_cam.setFromEuler(new THREE.Euler(Math.PI, 0.0, 0.0));
    R_world_threeworld.setFromEuler(new THREE.Euler(Math.PI / 2.0, 0.0, 0.0));
    const R_world_camera = R_world_threeworld.clone()
      .multiply(three_camera.quaternion)
      .multiply(R_threecam_cam);

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: three_camera.position
        .clone()
        .applyQuaternion(R_world_threeworld)
        .toArray(),
      aspect: three_camera.aspect,
      fov: (three_camera.fov * Math.PI) / 180.0,
    });
  }, [camera, sendCameraThrottled]);

  // What do we need to when the camera moves?
  sendCamera();
  const cameraChangedCallback = React.useCallback(() => {
    // If desired, send our camera via websocket.
    sendCamera();

    // Match all cameras.
    const globalCameras = viewer.globalCameras.current;
    if (globalCameras.synchronize) {
      globalCameras.synchronize = false;

      globalCameras.cameraControlRefs.forEach((other) => {
        if (cameraControlRef === other) return;
        const position = new THREE.Vector3();
        const target = new THREE.Vector3();
        cameraControlRef.current!.getPosition(position);
        cameraControlRef.current!.getTarget(target);
        other.current!.setLookAt(
          position.x,
          position.y,
          position.z,
          target.x,
          target.y,
          target.z
        );
      });

      // Hack to prevent the cameraChangedCallback() functions of other camera controls from firing.
      setTimeout(() => (globalCameras.synchronize = true), 1);
    }
  }, [viewer.globalCameras, camera, sendCamera]);

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = viewer.useGui((state) => state.websocketConnected);
  React.useEffect(() => {
    if (!connected) return;
    setTimeout(() => cameraChangedCallback(), 50);
  }, [connected, cameraChangedCallback]);

  React.useEffect(() => {
    const globalCameras = viewer.globalCameras.current;

    if (globalCameras.synchronize && globalCameras.cameras.length > 0) {
      const ours = cameraControlRef.current!;
      const other = globalCameras.cameraControlRefs[0].current!;
      const position = new THREE.Vector3();
      const target = new THREE.Vector3();
      other.getPosition(position);
      other.getTarget(target);
      ours.setLookAt(
        position.x,
        position.y,
        position.z,
        target.x,
        target.y,
        target.z
      );
    }

    globalCameras.cameras.push(camera);
    globalCameras.cameraControlRefs.push(cameraControlRef);

    window.addEventListener("resize", cameraChangedCallback);

    return () => {
      window.removeEventListener("resize", cameraChangedCallback);

      // Remove ourself from camera list. Since we always add/remove panels
      // from the end, a pop() would actually work as well here in constant
      // time.
      globalCameras.cameras.splice(globalCameras.cameras.indexOf(camera), 1);
      globalCameras.cameraControlRefs.splice(
        globalCameras.cameraControlRefs.indexOf(cameraControlRef),
        1
      );
    };
  }, [cameraChangedCallback, camera, viewer.globalCameras]);

  // Keyboard controls.
  React.useEffect(() => {
    const KEYCODE = {
      W: 87,
      A: 65,
      S: 83,
      D: 68,
      ARROW_LEFT: 37,
      ARROW_UP: 38,
      ARROW_RIGHT: 39,
      ARROW_DOWN: 40,
    };
    const cameraControls = cameraControlRef.current!;

    const wKey = new holdEvent.KeyboardKeyHold(KEYCODE.W, 20);
    const aKey = new holdEvent.KeyboardKeyHold(KEYCODE.A, 20);
    const sKey = new holdEvent.KeyboardKeyHold(KEYCODE.S, 20);
    const dKey = new holdEvent.KeyboardKeyHold(KEYCODE.D, 20);
    aKey.addEventListener("holding", (event) => {
      cameraControls.truck(-0.01 * event!.deltaTime, 0, false);
    });
    dKey.addEventListener("holding", (event) => {
      cameraControls.truck(0.01 * event!.deltaTime, 0, false);
    });
    wKey.addEventListener("holding", (event) => {
      cameraControls.forward(0.01 * event!.deltaTime, false);
    });
    sKey.addEventListener("holding", (event) => {
      cameraControls.forward(-0.01 * event!.deltaTime, false);
    });

    const leftKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_LEFT, 20);
    const rightKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_RIGHT, 20);
    const upKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_UP, 20);
    const downKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_DOWN, 20);
    leftKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        -0.1 * THREE.MathUtils.DEG2RAD * event!.deltaTime,
        0,
        true
      );
    });
    rightKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0.1 * THREE.MathUtils.DEG2RAD * event!.deltaTime,
        0,
        true
      );
    });
    upKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        -0.05 * THREE.MathUtils.DEG2RAD * event!.deltaTime,
        true
      );
    });
    downKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        0.05 * THREE.MathUtils.DEG2RAD * event!.deltaTime,
        true
      );
    });

    // It seems like we should be disposing some event listeners, but we don't get errors despite not doing this. Worth revisiting.
  });

  return (
    <CameraControls
      ref={cameraControlRef}
      minDistance={0.5}
      maxDistance={200.0}
      dollySpeed={0.3}
      smoothTime={0.0}
      draggingSmoothTime={0.0}
      onChange={cameraChangedCallback}
      makeDefault
    />
  );
}
