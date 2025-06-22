import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { IcosphereMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

/**
 * Component for rendering icosphere meshes
 */
export const IcosphereMesh = React.forwardRef<
  THREE.Mesh,
  IcosphereMessage & { children?: React.ReactNode }
>(function IcosphereMesh(
  { children, ...message },
  ref: React.ForwardedRef<THREE.Mesh>,
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
    return new THREE.IcosahedronGeometry(
      message.props.radius,
      message.props.subdivisions,
    );
  }, [message.props.radius, message.props.subdivisions]);

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
    <mesh
      ref={ref}
      geometry={geometry}
      material={material}
      castShadow={message.props.cast_shadow}
      receiveShadow={message.props.receive_shadow}
    >
      <OutlinesIfHovered alwaysMounted />
      {children}
    </mesh>
  );
});
