import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { BoxMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

let boxGeometry: THREE.BoxGeometry | null = null;

/**
 * Component for rendering box meshes
 */
export const BoxMesh = React.forwardRef<
  THREE.Group,
  BoxMessage & { children?: React.ReactNode }
>(function BoxMesh(
  { children, ...message },
  ref: React.ForwardedRef<THREE.Group>,
) {
  // Create material based on props.
  const material = React.useMemo(() => {
    return createStandardMaterial(message.props);
  }, [
    message.props.material,
    message.props.color,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
  ]);

  // Create box geometry only once.
  if (boxGeometry === null) {
    boxGeometry = new THREE.BoxGeometry(1.0, 1.0, 1.0);
  }

  // Clean up material when it changes.
  React.useEffect(() => {
    return () => {
      if (material) material.dispose();
    };
  }, [material]);

  return (
    <group ref={ref}>
      <mesh
        geometry={boxGeometry}
        scale={message.props.dimensions}
        material={material}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
      >
        <OutlinesIfHovered alwaysMounted />
      </mesh>
      {children}
    </group>
  );
});
