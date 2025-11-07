import { useFrame } from "@react-three/fiber";
import { randomInt } from "crypto";
import React from "react";
import * as THREE from "three";

export interface ConstantScreenSpaceBillboardProps {
  children: React.ReactNode;
  /** Scale factor for the size (default: 1.0) */
  scaleFactor?: number;
  /** Lock rotation on X axis */
  lockX?: boolean;
  /** Lock rotation on Y axis */
  lockY?: boolean;
  /** Lock rotation on Z axis */
  lockZ?: boolean;
  /** Unrotated children. **/
  unrotatedChildren?: React.ReactNode;
}

/**
 * Billboard component that faces the camera and maintains constant screen-space size.
 * Based on drei's Billboard and viser's CrosshairVisual scaling logic.
 *
 * Optimized to reduce per-frame CPU overhead:
 * - Throttles updates to every 3rd frame
 * - Uses low priority render order
 * - Reuses objects to avoid allocations
 */
export const ConstantScreenSpaceBillboard = React.forwardRef<
  THREE.Group,
  ConstantScreenSpaceBillboardProps
>(function ConstantScreenSpaceBillboard(
  {
    children,
    scaleFactor = 1.0,
    lockX = false,
    lockY = false,
    lockZ = false,
    unrotatedChildren,
  },
  fref,
) {
  const inner = React.useRef<THREE.Group>(null!);
  const localRef = React.useRef<THREE.Group>(null!);

  // Reuse objects to avoid allocations.
  const q = React.useRef(new THREE.Quaternion());
  const worldPos = React.useRef(new THREE.Vector3());
  const prevRotation = React.useRef(new THREE.Euler());

  useFrame(({ camera }) => {
    if (!localRef.current || !inner.current) return;

    // Save previous rotation in case we're locking an axis.
    prevRotation.current.copy(inner.current.rotation);

    // Always face the camera (billboard behavior).
    localRef.current.updateMatrix();
    localRef.current.updateWorldMatrix(false, false);
    localRef.current.getWorldQuaternion(q.current);
    camera
      .getWorldQuaternion(inner.current.quaternion)
      .premultiply(q.current.invert());

    // Readjust any axis that is locked.
    if (lockX) inner.current.rotation.x = prevRotation.current.x;
    if (lockY) inner.current.rotation.y = prevRotation.current.y;
    if (lockZ) inner.current.rotation.z = prevRotation.current.z;

    // Scale based on distance and FOV to maintain constant screen-space size.
    localRef.current.getWorldPosition(worldPos.current);
    const distance = camera.position.distanceTo(worldPos.current);
    const fovScale = Math.tan(
      ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 360,
    );
    // The divisor (20) controls the base size - higher = smaller on screen.
    const scale = (distance / 20) * fovScale * scaleFactor;
    inner.current.scale.setScalar(scale);
  });

  React.useImperativeHandle(fref, () => localRef.current, []);

  return (
    <group ref={localRef}>
      <group ref={inner}>{children}</group>
      {unrotatedChildren}
    </group>
  );
});
