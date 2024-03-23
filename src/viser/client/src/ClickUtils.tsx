import React from "react";
import * as THREE from "three";
import { ViewerContextContents } from "./App";

/** Turn a click event into a normalized device coordinate (NDC) vector.
 * Normalizes click coordinates to be between -1 and 1, with (0, 0) being the center of the screen.
 * Uses offsetX/Y, and clientWidth/Height to get the coordinates.
 */
export function clickToNDC(
  viewer: ViewerContextContents,
  event: React.PointerEvent<HTMLDivElement>,
): THREE.Vector2 {
  const mouseVector = new THREE.Vector2();
  mouseVector.x =
    2 *
      ((event.nativeEvent.offsetX + 0.5) /
        viewer.canvasRef.current!.clientWidth) -
    1;
  mouseVector.y =
    1 -
    2 *
      ((event.nativeEvent.offsetY + 0.5) /
        viewer.canvasRef.current!.clientHeight);
  return mouseVector;
}

/** Turn a click event to normalized OpenCV coordinate (NDC) vector.
 * Normalizes click coordinates to be between (0, 0) as upper-left corner,
 * and (1, 1) as lower-right corner, with (0.5, 0.5) being the center of the screen.
 * Uses offsetX/Y, and clientWidth/Height to get the coordinates.
 */
export function clickToOpenCV(
  viewer: ViewerContextContents,
  event: React.PointerEvent<HTMLDivElement>,
): THREE.Vector2 {
  const mouseVector = new THREE.Vector2();
  mouseVector.x =
    (event.nativeEvent.offsetX + 0.5) / viewer.canvasRef.current!.clientWidth;
  mouseVector.y =
    (event.nativeEvent.offsetY + 0.5) / viewer.canvasRef.current!.clientHeight;
  return mouseVector;
}

/** Given a normalized click (using `normalizeClick`), check if it is within the bounds of the canvas.
 */
export function isClickValid(mouseVector: THREE.Vector2): boolean {
  return (
    mouseVector.x < 1 &&
    mouseVector.x > -1 &&
    mouseVector.y < 1 &&
    mouseVector.y > -1
  );
}
