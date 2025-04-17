import { ViewerContext } from "./App";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls";
import { useFrame, useThree } from "@react-three/fiber";
import React, { useContext, useEffect, useRef, useCallback, useState } from "react";
import { PerspectiveCamera } from "three";
import * as THREE from "three";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { useThrottledMessageSender } from "./WebsocketFunctions";

// Interface for the shim object to satisfy CameraControls usage
interface CameraControlsShim {
  enabled: boolean;
  getTarget(target: THREE.Vector3): THREE.Vector3;
  setLookAt(posX: number, posY: number, posZ: number, targetX: number, targetY: number, targetZ: number, enableTransition?: boolean): void;
  updateCameraUp(): void;
  // We don't need truck, forward, etc. as keyboard handler is PLC specific
}

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const { camera, gl } = useThree();
  const plcRef = useRef<PointerLockControls | null>(null);
  const [isLocked, setIsLocked] = useState(false);

  // Store initial pose for reset - needs position and a calculated lookAt
  const initialCameraRef = useRef<{
    position: THREE.Vector3;
    lookAt: THREE.Vector3;
  } | null>(null);

  // --- Shim Implementation --- 
  // Use useRef for the shim itself so its identity is stable 
  // for assignment to viewer.cameraControlRef.current 
  const cameraControlShimRef = useRef<CameraControlsShim | null>(null);
  if (cameraControlShimRef.current === null) { // Initialize only once
    cameraControlShimRef.current = {
        enabled: true,
        getTarget: (target: THREE.Vector3): THREE.Vector3 => {
            // Calculate a point slightly in front of the camera
            const lookDirection = new THREE.Vector3();
            camera.getWorldDirection(lookDirection);
            target.copy(camera.position).add(lookDirection.multiplyScalar(1.0)); // look 1 unit ahead
            return target;
        },
        setLookAt: (
            posX: number, posY: number, posZ: number,
            targetX: number, targetY: number, targetZ: number,
            enableTransition?: boolean, // Ignored
        ) => {
            camera.position.set(posX, posY, posZ);
            // Use Matrix4 lookAt to get the correct orientation, then extract quaternion
            const lookAtMatrix = new THREE.Matrix4();
            lookAtMatrix.lookAt(
                camera.position, // eye
                new THREE.Vector3(targetX, targetY, targetZ), // target
                camera.up, // up (use camera's current up)
            );
            camera.quaternion.setFromRotationMatrix(lookAtMatrix);
            camera.updateMatrixWorld(true);
            // If PLC is active, setting camera directly might fight it.
            // Consider if plcRef.current.unlock() is needed here.
            sendCamera(); // Send update after setting pose
        },
        updateCameraUp: () => {
            // PointerLockControls manages camera.up implicitly
            // No explicit action needed here, unlike OrbitControls
        },
    };
  }
  // --- End Shim Implementation ---

  // --- Effect to Create/Manage PLC and Assign Shim to Context ---
  useEffect(() => {
    const controls = new PointerLockControls(camera, gl.domElement);
    plcRef.current = controls;

    const handleLock = () => setIsLocked(true);
    const handleUnlock = () => setIsLocked(false);
    controls.addEventListener('lock', handleLock);
    controls.addEventListener('unlock', handleUnlock);

    // Assign the SHIM object to the context ref
    (viewer.cameraControlRef as React.MutableRefObject<CameraControlsShim | null>).current = cameraControlShimRef.current;

    return () => {
      controls.removeEventListener('lock', handleLock);
      controls.removeEventListener('unlock', handleUnlock);
      (viewer.cameraControlRef as React.MutableRefObject<CameraControlsShim | null>).current = null;
      plcRef.current = null;
    };
  }, [camera, gl.domElement, viewer]); // Include shim ref if its creation depends on state/props

  const sendCameraThrottled = useThrottledMessageSender(20);

  // --- sendCamera: Use Shim's getTarget for look_at --- 
  const sendCamera = useCallback(() => {
      if (!cameraControlShimRef.current) return; // Ensure shim exists

      const T_world_threeworld = computeT_threeworld_world(viewer).invert();
      const T_world_threeworld_matrix = new THREE.Matrix4().copy(T_world_threeworld);
      const R_threecam_cam = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0.0, 0.0));
      const T_world_camera_matrix = new THREE.Matrix4()
          .copy(T_world_threeworld_matrix)
          .multiply(camera.matrixWorld)
          .multiply(new THREE.Matrix4().makeRotationFromQuaternion(R_threecam_cam));
      const t_world_camera = new THREE.Vector3();
      const R_world_camera = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      T_world_camera_matrix.decompose(t_world_camera, R_world_camera, scale);
      
      // Calculate lookAt using the shim's getTarget method, then transform
      const lookAtLocal = cameraControlShimRef.current.getTarget(new THREE.Vector3());
      const R_world_threeworld_quat = new THREE.Quaternion().setFromRotationMatrix(T_world_threeworld_matrix);
      const lookAtWorld = lookAtLocal.applyQuaternion(R_world_threeworld_quat);

      const up = camera.up.clone().applyQuaternion(R_world_threeworld_quat);

      // Store initial pose using current position and calculated lookAt if needed
      if (initialCameraRef.current === null) {
          const initialLookAt = new THREE.Vector3();
          cameraControlShimRef.current.getTarget(initialLookAt); // Get initial lookAt via shim
          initialCameraRef.current = { position: camera.position.clone(), lookAt: initialLookAt };
      }

      sendCameraThrottled({
          type: "ViewerCameraMessage",
          wxyz: [R_world_camera.w, R_world_camera.x, R_world_camera.y, R_world_camera.z],
          position: t_world_camera.toArray(),
          aspect: (camera as PerspectiveCamera).aspect,
          fov: THREE.MathUtils.degToRad((camera as PerspectiveCamera).fov),
          look_at: lookAtWorld.toArray(), // Send the calculated world lookAt
          up_direction: up.toArray(),
      });
  }, [camera, viewer, sendCameraThrottled]); // Removed cameraControlShimRef from deps as it's stable

  // --- resetCamera: Use Shim's setLookAt --- 
  const resetCamera = useCallback(() => {
      if (initialCameraRef.current && cameraControlShimRef.current) {
          // Use the shim's setLookAt method
          const initial = initialCameraRef.current;
          cameraControlShimRef.current.setLookAt(
              initial.position.x, initial.position.y, initial.position.z,
              initial.lookAt.x, initial.lookAt.y, initial.lookAt.z,
              false // Transition flag ignored by our shim
          );
          // setLookAt calls sendCamera internally now
      }
  }, []); // Dependencies: initialCameraRef, cameraControlShimRef (stable refs)

  // --- Assign context refs sendCameraRef and resetCameraViewRef ---
  useEffect(() => {
    viewer.sendCameraRef.current = sendCamera;
    viewer.resetCameraViewRef.current = resetCamera;
    return () => {
      viewer.sendCameraRef.current = null;
      viewer.resetCameraViewRef.current = null;
    };
  }, [viewer, sendCamera, resetCamera]);

  // --- Connection / Resize Effects (no change needed) ---
  const connected = viewer.useGui((state) => state.websocketConnected);
  useEffect(() => {
    if (!connected) return;
    setTimeout(() => {
      // Initial camera ref storage now happens within sendCamera
      sendCamera();
    }, 100);
  }, [connected, sendCamera]); // Removed camera, viewer dependencies as handled within sendCamera

  const canvas = viewer.canvasRef.current!;
  useEffect(() => {
      if (!canvas) return;
      const resizeObserver = new ResizeObserver(() => { sendCamera(); });
      resizeObserver.observe(canvas);
      return () => resizeObserver.disconnect();
  }, [canvas, sendCamera]);

  // --- PLC Change listener (no change needed) ---
  useEffect(() => {
    const controls = plcRef.current;
    if (!controls) return;
    const handleChange = () => { if (controls.isLocked) { sendCamera(); } };
    controls.addEventListener("change", handleChange);
    return () => { controls.removeEventListener("change", handleChange); };
  }, [sendCamera]);

  // --- Keyboard controls: Check shim enabled --- 
  // ... (move refs remain the same) ...
  const moveForward = useRef(false), moveBackward = useRef(false), moveLeft = useRef(false), moveRight = useRef(false), moveUp = useRef(false), moveDown = useRef(false);
  const moveSpeed = 1.0;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.tagName === 'BUTTON')) {
        return;
      }
      // Check shim enabled status
      if (!cameraControlShimRef.current?.enabled) return;

      switch (event.code) {
        case 'KeyW': moveForward.current = true; break;
        // ... other move keys ...
        case 'KeyS': moveBackward.current = true; break;
        case 'KeyA': moveLeft.current = true; break;
        case 'KeyD': moveRight.current = true; break;
        case 'KeyE': moveUp.current = true; break;
        case 'KeyQ': moveDown.current = true; break;
        case 'KeyL':
          if (plcRef.current && !plcRef.current.isLocked) {
            plcRef.current.lock();
          }
          break;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      switch (event.code) {
        case 'KeyW': moveForward.current = false; break;
        // ... other move keys ...
        case 'KeyS': moveBackward.current = false; break;
        case 'KeyA': moveLeft.current = false; break;
        case 'KeyD': moveRight.current = false; break;
        case 'KeyE': moveUp.current = false; break;
        case 'KeyQ': moveDown.current = false; break;
      }
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
    };
  }, []); // Shim ref is stable

  // --- useFrame: Check shim enabled --- 
  // ... (vector declarations remain the same) ...
  const direction = new THREE.Vector3(), right = new THREE.Vector3(), worldUp = new THREE.Vector3(0, 1, 0);

  useFrame((state, delta) => {
    // Check BOTH PointerLock active AND the shim enabled flag
    if (!plcRef.current?.isLocked || !cameraControlShimRef.current?.enabled) return;
    const actualSpeed = moveSpeed * delta;
    let moved = false;
    // ... (movement logic using camera.position remains the same) ...
    camera.getWorldDirection(direction); direction.y = 0; direction.normalize();
    right.crossVectors(worldUp, direction).normalize().negate();
    if (moveForward.current) { camera.position.addScaledVector(direction, actualSpeed); moved = true; }
    if (moveBackward.current) { camera.position.addScaledVector(direction, -actualSpeed); moved = true; }
    if (moveLeft.current) { camera.position.addScaledVector(right, -actualSpeed); moved = true; }
    if (moveRight.current) { camera.position.addScaledVector(right, actualSpeed); moved = true; }
    if (moveUp.current) { camera.position.addScaledVector(worldUp, actualSpeed); moved = true; }
    if (moveDown.current) { camera.position.addScaledVector(worldUp, -actualSpeed); moved = true; }

    if (moved) { sendCamera(); }
  });

  // Component renders nothing visually
  return null;
}
