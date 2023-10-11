import { ViewerContext } from "./App";
import { makeThrottledMessageSender } from "./WebsocketFunctions";
import { useThree } from "@react-three/fiber";
import React, { useContext } from "react";
import { PerspectiveCamera } from "three";
import * as THREE from "three";

export function ScenePointerControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);

  const sendClickThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    20,
  );
  React.useEffect(() => {
    const onMouseClick = (e: MouseEvent) => {
      // Don't send click events if the scene pointer events are disabled.
      if (viewer.scenePointerCallbackCount.current === 0) return;

      // Check that the mouse event happened inside the canvasRef.
      if (e.target !== viewer.canvasRef.current!) return;

      // clientX/Y are relative to the viewport, offsetX/Y are relative to the canvasRef.
      // clientX==offsetX if there is no titlebar, but clientX>offsetX if there is a titlebar.
      const mouseVector = new THREE.Vector2();
      mouseVector.x =
        2 * (e.offsetX / viewer.canvasRef.current!.clientWidth) - 1;
      mouseVector.y =
        1 - 2 * (e.offsetY / viewer.canvasRef.current!.clientHeight);
      if (
        mouseVector.x > 1 ||
        mouseVector.x < -1 ||
        mouseVector.y > 1 ||
        mouseVector.y < -1
      )
        return;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouseVector, camera);

      sendClickThrottled({
        type: "ScenePointerMessage",
        event_type: "click",
        ray_origin: [
          raycaster.ray.origin.x,
          -raycaster.ray.origin.z,
          raycaster.ray.origin.y,
        ],
        ray_direction: [
          raycaster.ray.direction.x,
          -raycaster.ray.direction.z,
          raycaster.ray.direction.y,
        ],
      });
    };
    window.addEventListener("click", onMouseClick);
    return () => {
      window.removeEventListener("click", onMouseClick);
    };
  }, [camera, sendClickThrottled]);

  return null;
}
