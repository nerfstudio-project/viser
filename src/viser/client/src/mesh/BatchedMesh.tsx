import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { BatchedMeshesMessage } from "../WebsocketMessages";
import { BatchedMeshManager, setupBatchedMesh } from "./BatchedMeshManager";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";

/**
 * Component for rendering batched/instanced meshes
 */
export const BatchedMesh = React.forwardRef<
  InstancedMesh2,
  BatchedMeshesMessage
>(function BatchedMesh(message, ref) {
  // Create persistent geometry and material
  const [material, setMaterial] = React.useState<THREE.Material>();
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  // Ref to store mesh manager for proper disposal
  const [meshManager, setMeshManager] = React.useState<BatchedMeshManager>();

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
    setMeshManager(setupBatchedMesh(
      geometry,
      material,
      numInstances,
      message.props.lod,
      message.props.cast_shadow,
    ));

    // Cleanup function to ensure proper disposal
    return () => {
      meshManager?.dispose();
    };
  }, [material, geometry, message.props.lod, message.props.cast_shadow]);

  // Create Float32Arrays once for positions and orientations
  const batched_positions = React.useMemo(
    () =>
      new Float32Array(
        message.props.batched_positions.buffer.slice(
          message.props.batched_positions.byteOffset,
          message.props.batched_positions.byteOffset +
            message.props.batched_positions.byteLength,
        ),
      ),
    [message.props.batched_positions],
  );

  const batched_wxyzs = React.useMemo(
    () =>
      new Float32Array(
        message.props.batched_wxyzs.buffer.slice(
          message.props.batched_wxyzs.byteOffset,
          message.props.batched_wxyzs.byteOffset +
            message.props.batched_wxyzs.byteLength,
        ),
      ),
    [message.props.batched_wxyzs],
  );

  // Handle updates to instance positions/orientations
  React.useEffect(() => {
    if (meshManager === undefined) return;

    // Update instance count if needed
    const newNumInstances = batched_positions.length / 3;
    meshManager.setInstanceCount(newNumInstances);

    // Update instance transforms - use the arrays we already created
    meshManager.updateInstances(batched_positions, batched_wxyzs);
  }, [batched_positions, batched_wxyzs, meshManager]);

  if (!meshManager) {
    return null;
  }

  return (
    <>
      <primitive ref={ref} object={meshManager.getMesh()} />
      {/* Add hover outline component for highlighting hovered instances */}
      {geometry && (
        <BatchedMeshHoverOutlines
          geometry={geometry}
          batched_positions={batched_positions}
          batched_wxyzs={batched_wxyzs}
        />
      )}
    </>
  );
});
