import { ViewerContext } from "./App";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls";
import { useFrame, useThree } from "@react-three/fiber";
import React, { useContext, useEffect, useRef, useCallback, useState } from "react";
import { PerspectiveCamera } from "three";
import * as THREE from "three";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { useThrottledMessageSender } from "./WebsocketFunctions";

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const { camera, gl } = useThree();
  const controlsRef = useRef<PointerLockControls | null>(null);
  const [isLocked, setIsLocked] = useState(false);

  useEffect(() => {
    const controls = new PointerLockControls(camera, gl.domElement);
    controlsRef.current = controls;

    const handleLock = () => setIsLocked(true);
    const handleUnlock = () => setIsLocked(false);
    controls.addEventListener('lock', handleLock);
    controls.addEventListener('unlock', handleUnlock);

    return () => {
      controls.removeEventListener('lock', handleLock);
      controls.removeEventListener('unlock', handleUnlock);
      controlsRef.current = null;
    };
  }, [camera, gl.domElement]);

  const sendCameraThrottled = useThrottledMessageSender(20);

  const initialCameraRef = useRef<{
    position: THREE.Vector3;
    quaternion: THREE.Quaternion;
  } | null>(null);

  const R_threecam_cam = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI, 0.0, 0.0),
  );
  const T_world_threeworld_matrix = new THREE.Matrix4();
  const T_world_camera_matrix = new THREE.Matrix4();
  const R_world_camera = new THREE.Quaternion();
  const t_world_camera = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const camera_direction = new THREE.Vector3();
  const lookAt = new THREE.Vector3();
  const up = new THREE.Vector3();

  const sendCamera = useCallback(() => {
    const T_world_threeworld = computeT_threeworld_world(viewer).invert();
    T_world_threeworld_matrix.copy(T_world_threeworld);

    T_world_camera_matrix
      .copy(T_world_threeworld_matrix)
      .multiply(camera.matrixWorld)
      .multiply(new THREE.Matrix4().makeRotationFromQuaternion(R_threecam_cam));

    T_world_camera_matrix.decompose(t_world_camera, R_world_camera, scale);

    camera_direction.set(0, 0, -1).applyQuaternion(R_world_camera);
    lookAt.copy(t_world_camera).add(camera_direction);

    up.set(0, 1, 0).applyQuaternion(R_world_camera);

    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        position: camera.position.clone(),
        quaternion: camera.quaternion.clone(),
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
      position: t_world_camera.toArray(),
      aspect: (camera as PerspectiveCamera).aspect,
      fov: THREE.MathUtils.degToRad((camera as PerspectiveCamera).fov),
      look_at: lookAt.toArray(),
      up_direction: up.toArray(),
    });
  }, [camera, sendCameraThrottled, viewer]);

  viewer.resetCameraViewRef.current = () => {
    if (initialCameraRef.current && controlsRef.current) {
      camera.position.copy(initialCameraRef.current.position);
      camera.quaternion.copy(initialCameraRef.current.quaternion);
      camera.updateMatrixWorld();
      sendCamera();
    }
  };

  const connected = viewer.useGui((state) => state.websocketConnected);
  useEffect(() => {
    viewer.sendCameraRef.current = sendCamera;
    if (!connected) return;
    setTimeout(() => {
      if (initialCameraRef.current === null) {
        initialCameraRef.current = {
          position: camera.position.clone(),
          quaternion: camera.quaternion.clone(),
        };
      }
      sendCamera();
    }, 100);
  }, [connected, sendCamera, camera]);

  const canvas = viewer.canvasRef.current!;
  useEffect(() => {
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);
    return () => resizeObserver.disconnect();
  }, [canvas, sendCamera]);

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    const handleChange = () => {
      if (controls.isLocked) {
        sendCamera();
      }
    };

    controls.addEventListener("change", handleChange);
    return () => {
      controls.removeEventListener("change", handleChange);
    };
  }, [sendCamera]);

  const moveForward = useRef(false);
  const moveBackward = useRef(false);
  const moveLeft = useRef(false);
  const moveRight = useRef(false);
  const moveUp = useRef(false);
  const moveDown = useRef(false);
  const moveSpeed = 1.0;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.tagName === 'BUTTON')) {
          return;
      }

      switch (event.code) {
        case 'KeyW': moveForward.current = true; break;
        case 'KeyS': moveBackward.current = true; break;
        case 'KeyA': moveLeft.current = true; break;
        case 'KeyD': moveRight.current = true; break;
        case 'KeyE': moveUp.current = true; break;
        case 'KeyQ': moveDown.current = true; break;
        case 'KeyL':
          if (controlsRef.current && !controlsRef.current.isLocked) {
            controlsRef.current.lock();
          }
          break;
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      switch (event.code) {
        case 'KeyW': moveForward.current = false; break;
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
  }, []);

  const direction = new THREE.Vector3();
  const right = new THREE.Vector3();
  const worldUp = new THREE.Vector3(0, 1, 0);

  useFrame((state, delta) => {
    if (!controlsRef.current?.isLocked) return;

    const actualSpeed = moveSpeed * delta;
    let moved = false;

    camera.getWorldDirection(direction);
    direction.y = 0;
    direction.normalize();

    right.crossVectors(worldUp, direction).normalize();
    right.negate();

    if (moveForward.current) {
      camera.position.addScaledVector(direction, actualSpeed);
      moved = true;
    }
    if (moveBackward.current) {
      camera.position.addScaledVector(direction, -actualSpeed);
      moved = true;
    }
    if (moveLeft.current) {
      camera.position.addScaledVector(right, -actualSpeed);
      moved = true;
    }
    if (moveRight.current) {
      camera.position.addScaledVector(right, actualSpeed);
      moved = true;
    }
    if (moveUp.current) {
      camera.position.addScaledVector(worldUp, actualSpeed);
      moved = true;
    }
    if (moveDown.current) {
      camera.position.addScaledVector(worldUp, -actualSpeed);
      moved = true;
    }

    if (moved) {
      sendCamera();
    }
  });

  return null;
}
