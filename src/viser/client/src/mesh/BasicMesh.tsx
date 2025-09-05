import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { MeshMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

/**
 * Component for rendering basic THREE.js meshes
 */
export const BasicMesh = React.forwardRef<
  THREE.Mesh,
  MeshMessage & { children?: React.ReactNode }
>(function BasicMesh(
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
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(
          message.props.vertices.buffer.slice(
            message.props.vertices.byteOffset,
            message.props.vertices.byteOffset +
              message.props.vertices.byteLength,
          ),
        ),
        3,
      ),
    );
    geometry.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(
          message.props.faces.buffer.slice(
            message.props.faces.byteOffset,
            message.props.faces.byteOffset + message.props.faces.byteLength,
          ),
        ),
        1,
      ),
    );
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }, [message.props.vertices.buffer, message.props.faces.buffer]);

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
      <OutlinesIfHovered
        enableCreaseAngle={
          geometry.attributes.position.count < 1024 &&
          geometry.boundingSphere!.radius > 0.1
        }
      />
      {children}
    </mesh>
  );
});
