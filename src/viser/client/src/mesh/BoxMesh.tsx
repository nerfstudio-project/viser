import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { BoxMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

/**
 * Component for rendering box meshes
 */
export const BoxMesh = React.forwardRef<THREE.Mesh, BoxMessage>(
  function BoxMesh(message, ref: React.ForwardedRef<THREE.Mesh>) {
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
      const [width, height, depth] = message.props.dimensions;
      return new THREE.BoxGeometry(width, height, depth);
    }, [message.props.dimensions]);

    // Clean up resources when component unmounts.
    React.useEffect(() => {
      return () => {
        if (material) material.dispose();
        if (geometry) geometry.dispose();
      };
    }, [material, geometry]);

    return (
      <mesh
        ref={ref}
        geometry={geometry}
        material={material}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
      >
        <OutlinesIfHovered alwaysMounted />
      </mesh>
    );
  },
);
