import React from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { HoverableContext } from "../HoverContext";
import { OutlinesMaterial } from "../Outlines";

/**
 * Props for BatchedMeshHoverOutlines component
 */
interface BatchedMeshHoverOutlinesProps {
  geometry: THREE.BufferGeometry;
  batched_positions: Float32Array;
  batched_wxyzs: Float32Array;
  meshTransform?: {
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    scale: THREE.Vector3;
  };
  // Function to compute batch index from instance index - needed for cases like InstancedAxes
  // where instanceId (from hover) doesn't match batched_positions indexing
  computeBatchIndexFromInstanceIndex?: (instanceId: number) => number;
}

/**
 * A reusable component that renders hover outlines for batched/instanced meshes
 * Shows a highlighted outline around the instance that is currently being hovered
 */
export const BatchedMeshHoverOutlines: React.FC<
  BatchedMeshHoverOutlinesProps
> = ({
  geometry,
  batched_positions,
  batched_wxyzs,
  meshTransform,
  computeBatchIndexFromInstanceIndex,
}) => {
  // Get hover state from context
  const hoveredRef = React.useContext(HoverableContext)!;

  // Create outline mesh reference
  const outlineRef = React.useRef<THREE.Mesh>(null);

  // Get rendering context for screen size
  const gl = useThree((state) => state.gl);
  const contextSize = React.useMemo(
    () => gl.getDrawingBufferSize(new THREE.Vector2()),
    [gl],
  );

  // Create outline geometry based on the original geometry using memoization
  const outlineGeometry = React.useMemo(() => {
    if (!geometry) return null;
    // Clone the geometry to create an independent copy for the outline
    return geometry.clone();
  }, [geometry]);

  // Create outline material with fixed properties
  const outlineMaterial = React.useMemo(() => {
    const material = new OutlinesMaterial({
      side: THREE.BackSide,
    });

    // Set fixed properties to match OutlinesIfHovered
    material.thickness = 10;
    material.color = new THREE.Color(0xfbff00); // Yellow highlight color
    material.opacity = 0.8;
    material.size = contextSize;
    material.transparent = true;
    material.screenspace = true; // Use screenspace for consistent thickness
    material.toneMapped = true;

    return material;
  }, [contextSize]);

  // Separate cleanup for geometry and material to handle dependency changes correctly
  // Clean up geometry when it changes or component unmounts
  React.useEffect(() => {
    return () => {
      if (outlineGeometry) {
        outlineGeometry.dispose();
      }
    };
  }, [outlineGeometry]);

  // Clean up material when it changes or component unmounts
  React.useEffect(() => {
    return () => {
      if (outlineMaterial) {
        outlineMaterial.dispose();
      }
    };
  }, [outlineMaterial]);

  // Update outline position based on hover state
  useFrame(() => {
    if (!outlineRef.current || !outlineGeometry || !hoveredRef) return;

    // Hide by default
    outlineRef.current.visible = false;

    // Check if we're hovering and have a valid instanceId
    if (
      hoveredRef.current.isHovered &&
      hoveredRef.current.instanceId !== null
    ) {
      // Get the instance ID from the hover state
      const hoveredInstanceId = hoveredRef.current.instanceId;

      // Calculate the actual batch index using the mapping function if provided
      const batchIndex = computeBatchIndexFromInstanceIndex
        ? computeBatchIndexFromInstanceIndex(hoveredInstanceId)
        : hoveredInstanceId; // Default is identity mapping

      // Only show outline if the batch index is valid
      if (batchIndex >= 0 && batchIndex < batched_positions.length / 3) {
        // Position the outline at the hovered instance
        outlineRef.current.position.set(
          batched_positions[batchIndex * 3 + 0],
          batched_positions[batchIndex * 3 + 1],
          batched_positions[batchIndex * 3 + 2],
        );

        // Set rotation to match the hovered instance
        outlineRef.current.quaternion.set(
          batched_wxyzs[batchIndex * 4 + 1], // x
          batched_wxyzs[batchIndex * 4 + 2], // y
          batched_wxyzs[batchIndex * 4 + 3], // z
          batched_wxyzs[batchIndex * 4 + 0], // w
        );

        // Apply mesh transform if provided (for GLB assets)
        if (meshTransform) {
          // Create instance matrix from batched data
          const instanceMatrix = new THREE.Matrix4().compose(
            outlineRef.current.position,
            outlineRef.current.quaternion,
            new THREE.Vector3(1, 1, 1),
          );

          // Create mesh transform matrix
          const transformMatrix = new THREE.Matrix4().compose(
            meshTransform.position,
            meshTransform.rotation,
            meshTransform.scale,
          );

          // Create final matrix by right-multiplying (match how it's done in ThreeAssets.tsx)
          const finalMatrix = instanceMatrix.clone().multiply(transformMatrix);

          // Decompose the final matrix into position, quaternion, scale
          const position = new THREE.Vector3();
          const quaternion = new THREE.Quaternion();
          const scale = new THREE.Vector3();
          finalMatrix.decompose(position, quaternion, scale);

          // Apply the decomposed transformation
          outlineRef.current.position.copy(position);
          outlineRef.current.quaternion.copy(quaternion);
          outlineRef.current.scale.copy(scale);
        }

        // Show the outline
        outlineRef.current.visible = true;
      }
    }
  });

  // This is now handled by the earlier cleanup effect

  if (!hoveredRef || !outlineGeometry) {
    return null;
  }

  return (
    <mesh
      ref={outlineRef}
      geometry={outlineGeometry}
      material={outlineMaterial}
      visible={false}
    />
  );
};
