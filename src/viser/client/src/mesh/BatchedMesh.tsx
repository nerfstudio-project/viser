import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { BatchedMeshesMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";
import { BatchedMeshManager, setupBatchedMesh } from "./BatchedMeshManager";

/**
 * Component for rendering batched/instanced meshes
 */
export const BatchedMesh = React.forwardRef<
  THREE.InstancedMesh,
  BatchedMeshesMessage
>(function BatchedMesh(message) {
  // Create persistent geometry and material
  const [material, setMaterial] = React.useState<THREE.Material>();
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  // Ref to store mesh manager for proper disposal
  const meshManagerRef = React.useRef<BatchedMeshManager | null>(null);

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

  // Create the instanced mesh once
  React.useEffect(() => {
    if (material === undefined || geometry === undefined) return;

    const numInstances =
      message.props.batched_positions.byteLength /
      (3 * Float32Array.BYTES_PER_ELEMENT);

    // Create new manager
    meshManagerRef.current = setupBatchedMesh(
      geometry,
      material,
      numInstances,
      message.props.lod,
      message.props.cast_shadow,
    );

    // Cleanup function to ensure proper disposal
    return () => {
      if (meshManagerRef.current) {
        meshManagerRef.current.dispose();
        meshManagerRef.current = null;
      }
    };
  }, [material, geometry, message.props.lod, message.props.cast_shadow]);

  // Handle updates to instance positions/orientations
  React.useEffect(() => {
    if (!meshManagerRef.current) return;

    const batched_positions = new Float32Array(
      message.props.batched_positions.buffer.slice(
        message.props.batched_positions.byteOffset,
        message.props.batched_positions.byteOffset +
          message.props.batched_positions.byteLength,
      ),
    );

    const batched_wxyzs = new Float32Array(
      message.props.batched_wxyzs.buffer.slice(
        message.props.batched_wxyzs.byteOffset,
        message.props.batched_wxyzs.byteOffset +
          message.props.batched_wxyzs.byteLength,
      ),
    );

    // Update instance count if needed
    const newNumInstances =
      message.props.batched_positions.byteLength /
      (3 * Float32Array.BYTES_PER_ELEMENT);
    meshManagerRef.current.setInstanceCount(newNumInstances);

    // Update instance transforms
    meshManagerRef.current.updateInstances(batched_positions, batched_wxyzs);
  }, [
    message.props.batched_positions.buffer,
    message.props.batched_wxyzs.buffer,
  ]);

  if (!meshManagerRef.current) {
    return null;
  }

  return (
    <>
      <primitive object={meshManagerRef.current.getMesh()} />
      <OutlinesIfHovered alwaysMounted />
    </>
  );
});
