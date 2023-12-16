import { ViewerContextContents } from "./App";
import * as THREE from "three";

/** Helper for computing the transformation between the three.js world and the
 * Python-exposed world frames. This is useful for things like switching
 * between +Y and +Z up directions for the world frame. */
export function computeT_threeworld_world(viewer: ViewerContextContents) {
  const wxyz = viewer.nodeAttributesFromName.current[""]!.wxyz!;
  const position = viewer.nodeAttributesFromName.current[""]!.position ?? [
    0, 0, 0,
  ];
  return new THREE.Matrix4()
    .makeRotationFromQuaternion(
      new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]),
    )
    .setPosition(position[0], position[1], position[2]);
}
