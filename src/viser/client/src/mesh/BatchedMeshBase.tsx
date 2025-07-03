import React, { useMemo, useEffect } from "react";
import * as THREE from "three";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { MeshoptSimplifier } from "meshoptimizer";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";

// Define the types of LOD settings.
// - "off": No LOD.
// - "auto": Built-in heuristic to compute LOD levels based on geometry
//   complexity.
// - [number, number][]: Array of [distance, simplification_ratio] pairs
//   where distance is the camera distance threshold and simplification_ratio
//   is the fraction of triangles to keep (0.0 to 1.0).
type LodSetting = "off" | "auto" | [number, number][];

// Helper function to create LODs for the mesh.
function createLODs(
  mesh: InstancedMesh2,
  geometry: THREE.BufferGeometry,
  material: THREE.Material | THREE.Material[],
  lod: LodSetting,
): { geometries: THREE.BufferGeometry[]; materials: THREE.Material[] } {
  if (!mesh || !geometry || lod === "off") {
    return { geometries: [], materials: [] };
  }

  // Calculate LOD settings.
  let ratios: number[] = [];
  let distances: number[] = [];

  if (lod === "auto") {
    // Automatic LOD settings based on geometry complexity.
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
  } else {
    // Use provided custom LOD settings.
    ratios = lod.map((pair) => pair[1]);
    distances = lod.map((pair) => pair[0]);
  }

  // Create the LOD levels.
  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];

  ratios.forEach((ratio, index) => {
    // Calculate target triangle count based on the ratio.
    const targetCount =
      Math.floor((geometry.index!.array.length * ratio) / 3) * 3;
    const lodGeometry = geometry.clone();

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
    const lodMaterial = Array.isArray(material)
      ? material.map((x) => x.clone())
      : material.clone();

    // Add this LOD level to the instanced mesh.
    mesh.addLOD(lodGeometry, lodMaterial, distances[index]);

    // Store the geometry and materials for proper disposal later.
    geometries.push(lodGeometry);
    if (Array.isArray(lodMaterial)) {
      materials.push(...lodMaterial);
    } else {
      materials.push(lodMaterial);
    }
  });
  return { geometries, materials };
}

/**
 * Shared base component for batched mesh rendering
 *
 * This component replaces BatchedMeshManager with a more React-friendly approach
 * using react-three-fiber's JSX components.
 */
export const BatchedMeshBase = React.forwardRef<
  InstancedMesh2,
  {
    // Data for instance positions and orientations.
    batched_positions: Uint8Array;
    batched_wxyzs: Uint8Array;
    batched_scales: Uint8Array | null;
    batched_colors: Uint8Array | null;

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
>(function BatchedMeshBase(props, ref) {
  // Store the mesh instance in state so effects can depend on it.
  const [mesh, setMesh] = React.useState<InstancedMesh2 | null>(null);

  // Forward the ref from the parent.
  React.useImperativeHandle(ref, () => mesh!, [mesh]);

  // Reusable objects for transform calculations.
  const tempPosition = useMemo(() => new THREE.Vector3(), []);
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), []);
  const tempScale = useMemo(() => new THREE.Vector3(1, 1, 1), []);

  // Create and manage InstancedMesh2 manually.
  useEffect(() => {
    // Create new InstancedMesh2.
    const newMesh = new InstancedMesh2(props.geometry, props.material, {
      capacity: 1,
    });

    // Set initial properties.
    if (props.scale) {
      if (typeof props.scale === "number") {
        newMesh.scale.setScalar(props.scale);
      } else if (Array.isArray(props.scale)) {
        newMesh.scale.set(...props.scale);
      } else {
        newMesh.scale.copy(props.scale);
      }
    }

    // Create LODs if needed.
    let lodGeometries: THREE.BufferGeometry[] = [];
    let lodMaterials: THREE.Material[] = [];

    if (props.lod !== "off") {
      const lods = createLODs(
        newMesh,
        props.geometry,
        props.material,
        props.lod,
      );
      lodGeometries = lods.geometries;
      lodMaterials = lods.materials;
    }

    // Update state with new mesh.
    setMesh(newMesh);

    return () => {
      // Cleanup on unmount or when dependencies change.
      newMesh.disposeBVH();
      newMesh.dispose();
      // Dispose LOD resources captured via closure.
      lodGeometries.forEach((geometry) => geometry.dispose());
      lodMaterials.forEach((material) => material.dispose());
    };
  }, [props.geometry, props.lod, props.material]); // Recreate when these change.

  // Update instances when positions or orientations change.
  useEffect(() => {
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
    const scalesView = props.batched_scales
      ? new DataView(
          props.batched_scales.buffer,
          props.batched_scales.byteOffset,
          props.batched_scales.byteLength,
        )
      : null;

    // Update all instances.
    mesh.updateInstances((obj, index) => {
      // Calculate byte offsets for reading float values.
      const posOffset = index * 3 * 4; // 3 floats, 4 bytes per float.
      const wxyzOffset = index * 4 * 4; // 4 floats, 4 bytes per float.
      const scaleOffset =
        props.batched_scales &&
        props.batched_scales.byteLength ===
          (props.batched_wxyzs.byteLength / 4) * 3
          ? index * 3 * 4 // Per-axis scaling: 3 floats, 4 bytes per float.
          : index * 4; // Uniform scaling: 1 float, 4 bytes per float.

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

      // Read scale value if available.
      if (scalesView) {
        // Check if we have per-axis scaling (N,3) or uniform scaling (N,).
        if (
          props.batched_scales!.byteLength ===
          (props.batched_wxyzs.byteLength / 4) * 3
        ) {
          // Per-axis scaling: read 3 floats.
          tempScale.set(
            scalesView.getFloat32(scaleOffset, true), // x scale.
            scalesView.getFloat32(scaleOffset + 4, true), // y scale.
            scalesView.getFloat32(scaleOffset + 8, true), // z scale.
          );
        } else {
          // Uniform scaling: read 1 float and apply to all axes.
          const scale = scalesView.getFloat32(scaleOffset, true);
          tempScale.setScalar(scale);
        }
      } else {
        tempScale.set(1, 1, 1);
      }

      // Apply to the instance.
      obj.position.copy(tempPosition);
      obj.quaternion.copy(tempQuaternion);
      obj.scale.copy(tempScale);
    });
  }, [
    props.batched_positions,
    props.batched_wxyzs,
    props.batched_scales,
    mesh,
  ]);

  // Update instances when colors change.
  React.useEffect(() => {
    if (mesh === null || props.batched_colors === null) return;
    for (let i = 0; i < mesh.instancesCount; i++) {
      let color;
      if (props.batched_colors.byteLength == 3) {
        color = new THREE.Color(
          props.batched_colors[0] / 255,
          props.batched_colors[1] / 255,
          props.batched_colors[2] / 255,
        );
      } else if (props.batched_colors.byteLength === mesh.instancesCount * 3) {
        color = new THREE.Color(
          props.batched_colors[i * 3] / 255,
          props.batched_colors[i * 3 + 1] / 255,
          props.batched_colors[i * 3 + 2] / 255,
        );
      } else {
        console.error(
          `Invalid batched_colors length: ${props.batched_colors.byteLength}, expected 3 or ${mesh.instancesCount * 3}`,
        );
        color = new THREE.Color(1, 1, 1); // Default to white.
      }
      mesh.setColorAt(i, color);
    }
  }, [props.batched_colors, mesh]);

  // Update shadow settings.
  useEffect(() => {
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
  }, [props.cast_shadow, props.receive_shadow, mesh]);

  // Return the mesh as a primitive and hover outlines if clickable.
  return (
    <>
      {mesh && <primitive object={mesh} />}
      {props.clickable && (
        <BatchedMeshHoverOutlines
          geometry={props.geometry}
          batched_positions={props.batched_positions}
          batched_wxyzs={props.batched_wxyzs}
          batched_scales={props.batched_scales}
        />
      )}
    </>
  );
});
