import { ViewerContext } from "./App";
import { makeThrottledMessageSender } from "./WebsocketFunctions";
import { CameraControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import * as holdEvent from "hold-event";
import React, { useContext, useRef } from "react";
import { PerspectiveCamera } from "three";
import * as THREE from "three";

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);

  const sendCameraThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    20,
  );

  // Helper for resetting camera poses.
  const initialCameraRef = useRef<{
    camera: PerspectiveCamera;
    lookAt: THREE.Vector3;
  } | null>(null);

  viewer.resetCameraViewRef.current = () => {
    viewer.cameraControlRef.current!.setLookAt(
      initialCameraRef.current!.camera.position.x,
      initialCameraRef.current!.camera.position.y,
      initialCameraRef.current!.camera.position.z,
      initialCameraRef.current!.lookAt.x,
      initialCameraRef.current!.lookAt.y,
      initialCameraRef.current!.lookAt.z,
      true,
    );
    viewer.cameraRef.current!.up.set(
      initialCameraRef.current!.camera.up.x,
      initialCameraRef.current!.camera.up.y,
      initialCameraRef.current!.camera.up.z,
    );
    viewer.cameraControlRef.current!.updateCameraUp();
  };

  // Callback for sending cameras.
  const sendCamera = React.useCallback(() => {
    const three_camera = camera;
    const camera_control = viewer.cameraControlRef.current;

    if (camera_control === null) {
      // Camera controls not yet ready, let's re-try later.
      setTimeout(sendCamera, 10);
      return;
    }

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const R_threecam_cam = new THREE.Quaternion();
    const R_world_threeworld = new THREE.Quaternion();
    R_threecam_cam.setFromEuler(new THREE.Euler(Math.PI, 0.0, 0.0));
    R_world_threeworld.setFromEuler(new THREE.Euler(Math.PI / 2.0, 0.0, 0.0));
    const R_world_camera = R_world_threeworld.clone()
      .multiply(three_camera.quaternion)
      .multiply(R_threecam_cam);

    const look_at = camera_control
      .getTarget(new THREE.Vector3())
      .applyQuaternion(R_world_threeworld);
    const up = three_camera.up.clone().applyQuaternion(R_world_threeworld);

    //Store initial camera values
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        camera: three_camera.clone(),
        lookAt: camera_control.getTarget(new THREE.Vector3()),
      };
    }

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
      look_at: [look_at.x, look_at.y, look_at.z],
      up_direction: [up.x, up.y, up.z],
    });
  }, [camera, sendCameraThrottled]);

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = viewer.useGui((state) => state.websocketConnected);
  React.useEffect(() => {
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  React.useEffect(() => {
    window.addEventListener("resize", sendCamera);
    return () => {
      window.removeEventListener("resize", sendCamera);
    };
  }, [camera]);

  // Note: This fires for double-click, which might not be available for mobile.
  // Maybe we should also support mobile via long-presses? Worth investigating.
  // Also, instead of double-click, we could consider an alt-click or meta-click.
  const sendClickThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    20,
  );
  React.useEffect(() => {
    const onMouseDouble = (e: MouseEvent) => {
      // check that the mouse event happened inside the canvasRef.
      if (e.target !== viewer.canvasRef.current!) return;

      // clientX/Y are relative to the viewport, offsetX/Y are relative to the canvasRef.
      // clientX==offsetX if there is no titlebar, but clientX>offsetX if there is a titlebar.
      const mouseVector = new THREE.Vector2();
      console.log(e);
      console.log(viewer.canvasRef.current!.clientWidth, viewer.canvasRef.current!.clientHeight);
      mouseVector.x = 2 * (e.offsetX / viewer.canvasRef.current!.clientWidth) - 1;
      mouseVector.y = 1 - 2 * (e.offsetY / viewer.canvasRef.current!.clientHeight);

      const mouse_in_scene = !(
        mouseVector.x > 1 ||
        mouseVector.x < -1 ||
        mouseVector.y > 1 ||
        mouseVector.y < -1
      );
      if (!mouse_in_scene) { return; } 

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouseVector, camera);

      console.log(e.offsetX, e.offsetY);

      sendClickThrottled({
        type: 'RayClickMessage',
        origin: [
          raycaster.ray.origin.x, 
          -raycaster.ray.origin.z, 
          raycaster.ray.origin.y
        ],
        direction: [
          raycaster.ray.direction.x, 
          -raycaster.ray.direction.z, 
          raycaster.ray.direction.y
        ],
      });
    };
    window.addEventListener('dblclick', onMouseDouble, false);
    return () => {
      window.removeEventListener('dblclick', onMouseDouble, false);
    };
  }, [camera, sendClickThrottled]);

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
      SPACE: " ",
      Q: 81,
      E: 69,
    };
    const cameraControls = viewer.cameraControlRef.current!;

    const wKey = new holdEvent.KeyboardKeyHold(KEYCODE.W, 20);
    const aKey = new holdEvent.KeyboardKeyHold(KEYCODE.A, 20);
    const sKey = new holdEvent.KeyboardKeyHold(KEYCODE.S, 20);
    const dKey = new holdEvent.KeyboardKeyHold(KEYCODE.D, 20);
    const qKey = new holdEvent.KeyboardKeyHold(KEYCODE.Q, 20);
    const eKey = new holdEvent.KeyboardKeyHold(KEYCODE.E, 20);

    // TODO: these event listeners are currently never removed, even if this
    // component gets unmounted.
    aKey.addEventListener("holding", (event) => {
      cameraControls.truck(-0.002 * event?.deltaTime, 0, true);
    });
    dKey.addEventListener("holding", (event) => {
      cameraControls.truck(0.002 * event?.deltaTime, 0, true);
    });
    wKey.addEventListener("holding", (event) => {
      cameraControls.forward(0.002 * event?.deltaTime, true);
    });
    sKey.addEventListener("holding", (event) => {
      cameraControls.forward(-0.002 * event?.deltaTime, true);
    });
    qKey.addEventListener("holding", (event) => {
      cameraControls.elevate(0.002 * event?.deltaTime, true);
    });
    eKey.addEventListener("holding", (event) => {
      cameraControls.elevate(-0.002 * event?.deltaTime, true);
    });

    const leftKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_LEFT, 20);
    const rightKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_RIGHT, 20);
    const upKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_UP, 20);
    const downKey = new holdEvent.KeyboardKeyHold(KEYCODE.ARROW_DOWN, 20);
    leftKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    rightKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    upKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });
    downKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });

    // TODO: we currently don't remove any event listeners. This is a bit messy
    // because KeyboardKeyHold attaches listeners directly to the
    // document/window; it's unclear if we can remove these.
    return () => {
      return;
    };
  }, [CameraControls]);

  return (
    <CameraControls
      ref={viewer.cameraControlRef}
      minDistance={0.1}
      maxDistance={200.0}
      dollySpeed={0.3}
      smoothTime={0.05}
      draggingSmoothTime={0.0}
      onChange={sendCamera}
      makeDefault
    />
  );
}
