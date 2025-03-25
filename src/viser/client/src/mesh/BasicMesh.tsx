import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { MeshMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

/**
 * Component for rendering basic THREE.js meshes
 */
export const BasicMesh = React.forwardRef<THREE.Mesh, MeshMessage>(
  function BasicMesh(message, ref) {
    // Setup material using memoization.
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

    // Clean up resources when component unmounts.
    React.useEffect(() => {
      return () => {
        material.dispose();
        geometry.dispose();
      };
    }, [material, geometry]);

    // This check is no longer needed with useMemo since it always returns a value

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
