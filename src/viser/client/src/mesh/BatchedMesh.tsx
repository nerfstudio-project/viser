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

  // Setup a basic material just once - we'll update all properties via direct methods
  // This dramatically improves performance by avoiding material recreation
  const material = React.useMemo(
    () => {
      // Create a basic material with neutral properties - all will be updated in useEffect
      return createStandardMaterial({
        material: "standard", // Will be updated if different
        color: [128, 128, 128], // Will be updated immediately
        wireframe: false, // Will be updated
        opacity: null, // Will be updated
        flat_shading: false, // Will be updated
        side: "front", // Will be updated
      });
    },
    [
      // No dependencies - we never want to recreate the material
      // All properties will be updated via updateMaterialProperties
    ],
  );

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

  // 1. Update instance transforms (positions and orientations)
  React.useEffect(() => {
    // Pass buffer views directly to avoid creating new arrays.
    meshManager.updateInstances(
      message.props.batched_positions,
      message.props.batched_wxyzs,
    );
  }, [
    meshManager,
    message.props.batched_positions,
    message.props.batched_wxyzs,
  ]);

  // 2. Update shadow settings
  React.useEffect(() => {
    meshManager.updateShadowSettings(
      message.props.cast_shadow,
      message.props.receive_shadow,
    );
  }, [meshManager, message.props.cast_shadow, message.props.receive_shadow]);

  // 3. Update material properties.
  React.useEffect(() => {
    meshManager.updateMaterialProperties({
      color: message.props.color,
      wireframe: message.props.wireframe,
      opacity: message.props.opacity,
      flatShading: message.props.flat_shading,
      side: {
        front: THREE.FrontSide,
        back: THREE.BackSide,
        double: THREE.DoubleSide,
      }[message.props.side],
      transparent: message.props.opacity !== null,
      materialType: message.props.material,
    });
  }, [
    meshManager,
    message.props.color,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
    message.props.material, // Added material type to dependencies
  ]);

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
          batched_positions={message.props.batched_positions} /* Raw bytes containing float32 position values */
          batched_wxyzs={message.props.batched_wxyzs} /* Raw bytes containing float32 quaternion values */
        />
      )}
    </>
  );
});
