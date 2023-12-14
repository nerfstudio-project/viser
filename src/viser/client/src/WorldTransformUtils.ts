import { ViewerContextContents } from "./App";
import * as THREE from "three";

/** Helper for computing the rotation between the three.js world and the
 * Python-exposed world frames. This is useful for things like switching
 * between +Y and +Z up directions for the world frame. */
export function getR_threeworld_world(viewer: ViewerContextContents) {
  const wxyz = viewer.nodeAttributesFromName.current[""]!.wxyz!;
  return new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
}

export function computeR_threeworld_world(
  newUpDirection: [number, number, number],
) {
  const threeUp = new THREE.Vector3(0, 1, 0);
  const newUp = new THREE.Vector3().fromArray(newUpDirection).normalize();

  const R_threeworld_world = new THREE.Quaternion().setFromUnitVectors(
    newUp,
    threeUp,
  );

  // If we set +Y to up, +X and +Z should face the camera.
  // If we set +Z to up, +X and +Y should face the camera.
  // etc.
  const forwardDir = new THREE.Vector3(1, 0, 1)
    .normalize()
    .applyQuaternion(R_threeworld_world.clone().invert());
  const absForwardDir = new THREE.Vector3(
    Math.abs(forwardDir.x),
    Math.abs(forwardDir.y),
    Math.abs(forwardDir.z),
  );
  R_threeworld_world.multiply(
    new THREE.Quaternion().setFromUnitVectors(absForwardDir, forwardDir),
  );
  return [
    R_threeworld_world.w,
    R_threeworld_world.x,
    R_threeworld_world.y,
    R_threeworld_world.z,
  ] as [number, number, number, number];
}
