import * as THREE from "three";
import { ViewerContextContents } from "./ViewerContext";

/** Turn a click event into a normalized device coordinate (NDC) vector.
 * Normalizes click coordinates to be between -1 and 1, with (0, 0) being the center of the screen.
 *
 * Returns null if input is not valid.
 */
export function ndcFromPointerXy(
  viewer: ViewerContextContents,
  xy: [number, number],
): THREE.Vector2 | null {
  const mouseVector = new THREE.Vector2();
  mouseVector.x =
    2 * ((xy[0] + 0.5) / viewer.refs.current.canvas!.clientWidth) - 1;
  mouseVector.y =
    1 - 2 * ((xy[1] + 0.5) / viewer.refs.current.canvas!.clientHeight);
  return mouseVector.x < 1 &&
    mouseVector.x > -1 &&
    mouseVector.y < 1 &&
    mouseVector.y > -1
    ? mouseVector
    : null;
}

/** Turn a click event to normalized OpenCV coordinate (NDC) vector.
 * Normalizes click coordinates to be between (0, 0) as upper-left corner,
 * and (1, 1) as lower-right corner, with (0.5, 0.5) being the center of the screen.
 * Uses offsetX/Y, and clientWidth/Height to get the coordinates.
 */
export function opencvXyFromPointerXy(
  viewer: ViewerContextContents,
  xy: [number, number],
): THREE.Vector2 {
  const mouseVector = new THREE.Vector2();
  mouseVector.x = (xy[0] + 0.5) / viewer.refs.current.canvas!.clientWidth;
  mouseVector.y = (xy[1] + 0.5) / viewer.refs.current.canvas!.clientHeight;
  return mouseVector;
}
