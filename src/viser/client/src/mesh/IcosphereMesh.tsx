import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { IcosphereMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

// Cache icosphere geometries based on # of subdivisions. In theory this cache
// can grow indefinitely, but this doesn't seem worth the complexity of
// preventing.
const icosphereGeometryCache = new Map<number, THREE.IcosahedronGeometry>();

/**
 * Component for rendering icosphere meshes
 */
export const IcosphereMesh = React.forwardRef<
  THREE.Group,
  IcosphereMessage & { children?: React.ReactNode }
>(function IcosphereMesh(
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

  // Setup geometry using memoization.
  const geometry = React.useMemo(() => {
    if (!icosphereGeometryCache.has(message.props.subdivisions)) {
      icosphereGeometryCache.set(
        message.props.subdivisions,
        new THREE.IcosahedronGeometry(1.0, message.props.subdivisions),
      );
    }
    return icosphereGeometryCache.get(message.props.subdivisions)!;
  }, [message.props.subdivisions]);

  // Clean up geometry when it changes.
  React.useEffect(() => {
    return () => {
      if (geometry) geometry.dispose();
    };
  }, [geometry]);

  // Clean up material when it changes.
  React.useEffect(() => {
    return () => {
      if (material) material.dispose();
    };
  }, [material]);

  return (
    <group ref={ref}>
      <mesh
        ref={ref}
        geometry={geometry}
        scale={message.props.radius}
        material={material}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
      >
        <OutlinesIfHovered />
      </mesh>
      {children}
    </group>
  );
});
