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
  /** Raw bytes containing float32 position values (xyz) */
  batched_positions: Uint8Array;
  /** Raw bytes containing float32 quaternion values (wxyz) */
  batched_wxyzs: Uint8Array;
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
// Static reusable objects for matrix operations.
const _tempObjects = {
  instanceMatrix: new THREE.Matrix4(),
  transformMatrix: new THREE.Matrix4(),
  finalMatrix: new THREE.Matrix4(),
  position: new THREE.Vector3(),
  quaternion: new THREE.Quaternion(),
  scale: new THREE.Vector3(),
  oneVector: new THREE.Vector3(1, 1, 1),
};

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

      // Create DataViews to read float values
      const positionsView = new DataView(
        batched_positions.buffer,
        batched_positions.byteOffset,
        batched_positions.byteLength,
      );

      const wxyzsView = new DataView(
        batched_wxyzs.buffer,
        batched_wxyzs.byteOffset,
        batched_wxyzs.byteLength,
      );

      // Only show outline if the batch index is valid (check bytes per position = 3 floats * 4 bytes)
      if (batchIndex >= 0 && batchIndex * 12 < batched_positions.byteLength) {
        // Calculate byte offsets.
        const posOffset = batchIndex * 3 * 4; // 3 floats, 4 bytes per float
        const wxyzOffset = batchIndex * 4 * 4; // 4 floats, 4 bytes per float

        // Position the outline at the hovered instance
        outlineRef.current.position.set(
          positionsView.getFloat32(posOffset, true), // x
          positionsView.getFloat32(posOffset + 4, true), // y
          positionsView.getFloat32(posOffset + 8, true), // z
        );

        // Set rotation to match the hovered instance
        outlineRef.current.quaternion.set(
          wxyzsView.getFloat32(wxyzOffset + 4, true), // x
          wxyzsView.getFloat32(wxyzOffset + 8, true), // y
          wxyzsView.getFloat32(wxyzOffset + 12, true), // z
          wxyzsView.getFloat32(wxyzOffset, true), // w
        );

        // Apply mesh transform if provided (for GLB assets)
        if (meshTransform) {
          // Create instance matrix from batched data.
          _tempObjects.instanceMatrix.compose(
            outlineRef.current.position,
            outlineRef.current.quaternion,
            _tempObjects.oneVector,
          );

          // Create mesh transform matrix.
          _tempObjects.transformMatrix.compose(
            meshTransform.position,
            meshTransform.rotation,
            meshTransform.scale,
          );

          // Create final matrix by right-multiplying (match how it's done in ThreeAssets.tsx).
          _tempObjects.finalMatrix
            .copy(_tempObjects.instanceMatrix)
            .multiply(_tempObjects.transformMatrix);

          // Decompose the final matrix into position, quaternion, scale.
          _tempObjects.finalMatrix.decompose(
            _tempObjects.position,
            _tempObjects.quaternion,
            _tempObjects.scale,
          );

          // Apply the decomposed transformation.
          outlineRef.current.position.copy(_tempObjects.position);
          outlineRef.current.quaternion.copy(_tempObjects.quaternion);
          outlineRef.current.scale.copy(_tempObjects.scale);
        }

        // Show the outline.
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
