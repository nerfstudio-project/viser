import React, { useRef, useMemo, useEffect } from "react";
import * as THREE from "three";
import { extend, ThreeElement } from "@react-three/fiber";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { MeshoptSimplifier } from "meshoptimizer";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";

// Add InstancedMesh2 to the jsx catalog.
extend({ InstancedMesh2 });

declare module "@react-three/fiber" {
  interface ThreeElements {
    instancedMesh2: ThreeElement<typeof InstancedMesh2>;
  }
}

// Define the types of LOD settings.
type LodSetting = "off" | "auto" | [number, number][];

/**
 * Props for the shared BatchedMeshBase component
 */
export interface BatchedMeshBaseProps {
  // Data for instance positions and orientations.
  batched_positions: Uint8Array;
  batched_wxyzs: Uint8Array;

  // Geometry info.
  geometry: THREE.BufferGeometry;

  // Material info.
  material: THREE.Material | THREE.Material[];

  // Rendering options.
  lod: LodSetting;
  cast_shadow: boolean;
  receive_shadow: boolean;

  // Optional props.
  scale?: THREE.Vector3 | [number, number, number] | number;
  clickable?: boolean;
}

/**
 * Shared base component for batched mesh rendering
 *
 * This component replaces BatchedMeshManager with a more React-friendly approach
 * using react-three-fiber's JSX components.
 */
export const BatchedMeshBase = React.forwardRef<
  InstancedMesh2,
  BatchedMeshBaseProps
>(function BatchedMeshBase(props, ref) {
  // Create refs for our meshes.
  const instancedMeshRef = useRef<InstancedMesh2>(null);

  // Forward the ref from the parent.
  React.useImperativeHandle(ref, () => instancedMeshRef.current!, []);

  // Store LOD meshes.
  const [lodGeometries, setLodGeometries] = React.useState<
    THREE.BufferGeometry[]
  >([]);

  // Reusable objects for transform calculations.
  const tempPosition = useMemo(() => new THREE.Vector3(), []);
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), []);
  const tempScale = useMemo(() => new THREE.Vector3(1, 1, 1), []);

  // Set up the initial instances.
  useEffect(() => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    // Create LODs if needed.
    if (props.lod !== "off") {
      createLODs();
    }

    return () => {
      // Clean up LOD geometries when component unmounts.
      lodGeometries.forEach((geometry) => geometry.dispose());

      if (mesh) {
        mesh.disposeBVH();
      }
    };
  }, [props.geometry, props.lod]);

  // Update instances when positions or orientations change.
  useEffect(() => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    const instanceCount =
      props.batched_positions.byteLength / (3 * Float32Array.BYTES_PER_ELEMENT);
    if (mesh.instancesCount !== instanceCount) {
      if (mesh.capacity < instanceCount) {
        // Increase capacity if needed.
        mesh.resizeBuffers(instanceCount);
      }
      mesh.clearInstances();
      mesh.addInstances(instanceCount, () => {});
      mesh.computeBVH();
    }

    // Create views to efficiently read float values.
    const positionsView = new DataView(
      props.batched_positions.buffer,
      props.batched_positions.byteOffset,
      props.batched_positions.byteLength,
    );

    const wxyzsView = new DataView(
      props.batched_wxyzs.buffer,
      props.batched_wxyzs.byteOffset,
      props.batched_wxyzs.byteLength,
    );

    // Update all instances.
    mesh.updateInstances((obj, index) => {
      // Calculate byte offsets for reading float values.
      const posOffset = index * 3 * 4; // 3 floats, 4 bytes per float.
      const wxyzOffset = index * 4 * 4; // 4 floats, 4 bytes per float.

      // Read position values.
      tempPosition.set(
        positionsView.getFloat32(posOffset, true), // x.
        positionsView.getFloat32(posOffset + 4, true), // y.
        positionsView.getFloat32(posOffset + 8, true), // z.
      );

      // Read quaternion values.
      tempQuaternion.set(
        wxyzsView.getFloat32(wxyzOffset + 4, true), // x.
        wxyzsView.getFloat32(wxyzOffset + 8, true), // y.
        wxyzsView.getFloat32(wxyzOffset + 12, true), // z.
        wxyzsView.getFloat32(wxyzOffset, true), // w (first value).
      );

      // Apply to the instance.
      obj.position.copy(tempPosition);
      obj.quaternion.copy(tempQuaternion);
      obj.scale.copy(tempScale);
    });
  }, [props.batched_positions, props.batched_wxyzs]);

  // Update shadow settings.
  useEffect(() => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    mesh.castShadow = props.cast_shadow;
    mesh.receiveShadow = props.receive_shadow;

    // Update all LOD objects too.
    if (mesh.LODinfo && mesh.LODinfo.objects) {
      mesh.LODinfo.objects.forEach((obj) => {
        obj.castShadow = props.cast_shadow;
        obj.receiveShadow = props.receive_shadow;
      });
    }
  }, [props.cast_shadow, props.receive_shadow]);

  // Helper function to create LODs for the mesh.
  const createLODs = React.useCallback(() => {
    const mesh = instancedMeshRef.current;
    if (!mesh || !props.geometry) return;

    // Clear any existing LODs.
    lodGeometries.forEach((geometry) => geometry.dispose());
    setLodGeometries([]);

    // Create dummy mesh for LOD calculations.

    // Calculate LOD settings.
    let ratios: number[] = [];
    let distances: number[] = [];

    if (props.lod === "auto") {
      // Automatic LOD settings based on geometry complexity.
      const geometry = props.geometry;
      geometry.computeBoundingSphere();
      const boundingRadius = geometry.boundingSphere!.radius;
      const vertexCount = geometry.attributes.position.count;

      // 1. Compute LOD ratios based on vertex count.
      if (vertexCount > 10_000) {
        ratios = [0.2, 0.05, 0.01]; // Very complex
      } else if (vertexCount > 2_000) {
        ratios = [0.4, 0.1, 0.03]; // Medium complex
      } else if (vertexCount > 500) {
        ratios = [0.6, 0.2, 0.05]; // Light
      } else {
        ratios = [0.85, 0.4, 0.1]; // Already simple
      }

      // 2. Compute LOD distances based on bounding radius.
      const sizeFactor = Math.sqrt(boundingRadius + 1e-5);
      const baseMultipliers = [1, 2, 3]; // Distance "steps" for LOD switching.
      distances = baseMultipliers.map((m) => m * sizeFactor);
    } else if (props.lod !== "off") {
      // Use provided custom LOD settings.
      ratios = props.lod.map((pair: [number, number]) => pair[1]);
      distances = props.lod.map((pair: [number, number]) => pair[0]);
    }

    // Create the LOD levels.
    const newLodGeometries: THREE.BufferGeometry[] = [];

    ratios.forEach((ratio, index) => {
      // Calculate target triangle count based on the ratio.
      const targetCount =
        Math.floor((props.geometry.index!.array.length * ratio) / 3) * 3;
      const lodGeometry = props.geometry.clone();

      // Use meshopt to simplify the geometry.
      const dstIndexArray = MeshoptSimplifier.simplify(
        new Uint32Array(lodGeometry.index!.array),
        new Float32Array(lodGeometry.attributes.position.array),
        3,
        targetCount,
        0.01, // Error tolerance.
        ["LockBorder"], // Prevents triangle flipping artifacts.
      )[0];

      // Update the geometry with the simplified version.
      lodGeometry.index!.array.set(dstIndexArray);
      lodGeometry.index!.needsUpdate = true;
      lodGeometry.setDrawRange(0, dstIndexArray.length);

      // Create a cloned material for this LOD level.
      const lodMaterial = Array.isArray(props.material)
        ? props.material.map((x) => x.clone())
        : props.material.clone();

      // Add this LOD level to the instanced mesh.
      mesh.addLOD(lodGeometry, lodMaterial, distances[index]);

      // Store the geometry for proper disposal later.
      newLodGeometries.push(lodGeometry);
    });

    setLodGeometries(newLodGeometries);
  }, [props.lod, props.geometry]);

  // We want to use the same object during each remount to prevent
  // instancedMesh2 from re-initializing.
  const [meshParams] = React.useState(() => ({ capacity: 1 }));

  // Handle click events for the instanced mesh.
  return (
    <>
      <instancedMesh2
        ref={instancedMeshRef}
        args={[props.geometry, props.material, meshParams]}
        castShadow={props.cast_shadow}
        receiveShadow={props.receive_shadow}
        scale={props.scale}
      />

      {/* Add hover outline component for highlighting hovered instances */}
      {props.clickable && props.geometry && (
        <BatchedMeshHoverOutlines
          geometry={props.geometry}
          batched_positions={props.batched_positions}
          batched_wxyzs={props.batched_wxyzs}
        />
      )}
    </>
  );
});
