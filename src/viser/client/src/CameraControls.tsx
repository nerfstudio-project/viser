import { ViewerContext } from "./App";
import { makeThrottledMessageSender } from "./WebsocketFunctions";
import { PointerLockControls, OrbitControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import React, { useContext, useEffect, useRef, useCallback, useState } from "react";
import * as THREE from "three";



export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const [isFirstPerson, setIsFirstPerson] = useState(false);


  const { camera, gl } = useThree();
  const controlsRef = useRef(null);
  const orbitControlsRef = useRef(null);
  const speed = 0.03;

  const sendCameraThrottled = makeThrottledMessageSender(
    viewer,
    20,
  );

  // helper for resetting camera poses.
  const initialCameraRef = useRef<{
    position: THREE.Vector3;
    rotation: THREE.Euler;
  } | null>(null);

  viewer.resetCameraViewRef.current = () => {
    if (initialCameraRef.current) {
      camera.position.copy(initialCameraRef.current.position);
      camera.rotation.copy(initialCameraRef.current.rotation);
    }
  };

  // Callback for sending cameras.
  const sendCamera = useCallback(() => {
    if (!controlsRef.current && !orbitControlsRef.current) return;

    const { position, quaternion } = camera;
    const rotation = new THREE.Euler().setFromQuaternion(quaternion);

    // Store initial camera values
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        position: position.clone(),
        rotation: rotation.clone(),
      };
    }

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
      position: position.toArray(),
      aspect: (camera as THREE.PerspectiveCamera).aspect || 1,
      fov: ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0 || 0,
      look_at: [0, 0, 0], // Not used in first-person view
      up_direction: [camera.up.x, camera.up.y, camera.up.z],
    });
  }, [camera, sendCameraThrottled]);

  // new connections.
  const connected = viewer.useGui((state) => state.websocketConnected);
  useEffect(() => {
    viewer.sendCameraRef.current = sendCamera;
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  // Send camera for 3D viewport changes.
  const canvas = viewer.canvasRef.current!; // R3F canvas.
  useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);

    // clean up .
    return () => resizeObserver.disconnect();
  }, [canvas]);

  // state for the for camera velocity
  const [velocity, setVelocity] = useState(new THREE.Vector3());

  // Apply velocity to the camera
  useEffect(() => {
    const applyVelocity = () => {
      camera.translateX(velocity.x);
      camera.translateY(velocity.y);
      camera.translateZ(velocity.z);
      sendCamera();

      // ~apply damping to simulate inertia
      velocity.multiplyScalar(0.9);

      // Stop the loop if velocity is very small
      if (velocity.length() > 0.001) {
        requestAnimationFrame(applyVelocity);
      }
    };

    applyVelocity();
  }, [velocity, camera, sendCamera]);

  // Keyboard controls for movement.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const newVelocity = velocity.clone();
      switch (event.key) {
        case 'w':
          newVelocity.z -= speed;
          break;
        case 's':
          newVelocity.z += speed;
          break;
        case 'a':
          newVelocity.x -= speed;
          break;
        case 'd':
          newVelocity.x += speed;
          break;
        case 'q':
          newVelocity.y -= speed;
          break;
        case 'e':
          newVelocity.y += speed;
          break;
        case 'p':
        
          setIsFirstPerson(prev => {
            if (prev) {
              // If switching from first-person to orbit, release the pointer lock
              document.exitPointerLock();
            }
            return !prev;});
        break;
        default:
          break;
      }
      setVelocity(newVelocity);
    };

    window.addEventListener('keydown', handleKeyDown);

    // Cleanup event listener on component unmount
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [velocity]);

  return (
    <>
    {isFirstPerson ? (
      <PointerLockControls ref={controlsRef} args={[camera, gl.domElement]} />
    ) : (
      <OrbitControls ref={orbitControlsRef} args={[camera, gl.domElement]} />
    )}
  </>
  );
}
