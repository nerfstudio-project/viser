import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { BatchedMeshesMessage } from "../WebsocketMessages";
import { BatchedMeshManager } from "./BatchedMeshManager";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";
import { ViewerContext } from "../ViewerContext";

/**
 * Component for rendering batched/instanced meshes
 */
export const BatchedMesh = React.forwardRef<
  InstancedMesh2,
  BatchedMeshesMessage
>(function BatchedMesh(message, ref) {
  const viewer = React.useContext(ViewerContext)!;
  const clickable =
    viewer.useSceneTree(
      (state) => state.nodeFromName[message.name]?.clickable,
    ) ?? false;

  // Setup material using memoization, but without color in dependencies
  // We'll update color separately via updateMaterialColor for better performance
  const material = React.useMemo(() => {
    // Create material with an initial color that will be immediately updated in useEffect
    return createStandardMaterial({
      ...message.props,
      color: [128, 128, 128] // Initial neutral gray, will be updated
    });
  }, [
    // Dependencies excluding color (handled separately for performance)
    message.props.material,
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

  // Create Float32Arrays once for positions and orientations.
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

  // Create mesh manager with useMemo for better performance.
  const meshManager = React.useMemo(() => {
    const numInstances =
      message.props.batched_positions.byteLength /
      (3 * Float32Array.BYTES_PER_ELEMENT);

    // Create new manager without shadow settings to reduce dependencies
    return new BatchedMeshManager(
      geometry,
      material,
      numInstances,
      message.props.lod,
    );
  }, [
    geometry,
    material,
    message.props.lod,
    message.props.batched_positions.byteLength, // Keep this to handle instance count changes
  ]);

  // Effects for properties that can be updated without recreating the mesh
  // We use separate effects to minimize unnecessary work when only some props change
  
  // 1. Update instance transforms (positions and orientations)
  React.useEffect(() => {
    meshManager.updateInstances(batched_positions, batched_wxyzs);
  }, [meshManager, batched_positions, batched_wxyzs]);
  
  // 2. Update shadow settings
  React.useEffect(() => {
    meshManager.updateShadowSettings(
      message.props.cast_shadow,
      message.props.receive_shadow
    );
  }, [meshManager, message.props.cast_shadow, message.props.receive_shadow]);
  
  // 3. Update material color
  React.useEffect(() => {
    if (message.props.color) {
      meshManager.updateMaterialColor(message.props.color);
    }
  }, [meshManager, message.props.color]);

  // Handle cleanup when dependencies change or component unmounts.
  React.useEffect(() => {
    return () => {
      meshManager.dispose();
    };
  }, [meshManager]);

  return (
    <>
      <primitive ref={ref} object={meshManager.getMesh()} />
      {/* Add hover outline component for highlighting hovered instances */}
      {clickable && geometry && (
        <BatchedMeshHoverOutlines
          geometry={geometry}
          batched_positions={batched_positions}
          batched_wxyzs={batched_wxyzs}
        />
      )}
    </>
  );
});
