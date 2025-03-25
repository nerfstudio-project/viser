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
    // Create persistent geometry and material
    const [material, setMaterial] = React.useState<THREE.Material>();
    const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();

    // Setup material
    React.useEffect(() => {
      const material = createStandardMaterial(message.props);
      setMaterial(material);

      return () => {
        // Dispose material when done
        material.dispose();
      };
    }, [
      message.props.material,
      message.props.color,
      message.props.wireframe,
      message.props.opacity,
      message.props.flat_shading,
      message.props.side,
    ]);

    // Setup geometry
    React.useEffect(() => {
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

      setGeometry(geometry);

      return () => {
        geometry.dispose();
      };
    }, [message.props.vertices.buffer, message.props.faces.buffer]);

    if (geometry === undefined || material === undefined) {
      return null;
    }

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
