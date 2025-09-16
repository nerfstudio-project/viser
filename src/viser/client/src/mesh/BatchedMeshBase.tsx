import React, { useMemo, useEffect, useRef } from "react";
import * as THREE from "three";
import { MeshoptSimplifier } from "meshoptimizer";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";
import { useThree, useFrame } from "@react-three/fiber";

// Define the types of LOD settings.
// - "off": No LOD.
// - "auto": Built-in heuristic to compute LOD levels based on geometry
//   complexity.
// - [number, number][]: Array of [distance, simplification_ratio] pairs
//   where distance is the camera distance threshold and simplification_ratio
//   is the fraction of triangles to keep (0.0 to 1.0).
type LodSetting = "off" | "auto" | [number, number][];

// LOD level info
interface LODLevel {
  mesh: THREE.InstancedMesh;
  geometry: THREE.BufferGeometry;
  material: THREE.Material | THREE.Material[];
  distance: number;
  ratio: number;
}

// Helper to compute LOD settings
function computeLODSettings(
  geometry: THREE.BufferGeometry,
  lod: LodSetting,
): { ratios: number[]; distances: number[] } {
  if (lod === "off") {
    return { ratios: [], distances: [] };
  }

  geometry.computeBoundingSphere();

  if (lod === "auto") {
    const boundingRadius = geometry.boundingSphere!.radius;
    const vertexCount = geometry.attributes.position.count;

    // Skip LOD for very simple meshes
    if (vertexCount < 50) {
      return { ratios: [], distances: [] };
    }

    let ratios: number[];
    if (vertexCount < 500) {
      ratios = [0.6, 0.2];
    } else if (vertexCount < 2000) {
      ratios = [0.4, 0.1, 0.03];
    } else {
      ratios = [0.2, 0.05, 0.01];
    }

    const sizeFactor = Math.sqrt(boundingRadius + 1e-5);
    const baseMultipliers = [1, 2, 3].slice(0, ratios.length);
    const distances = baseMultipliers.map((m) => m * sizeFactor);

    return { ratios, distances };
  } else {
    return {
      ratios: lod.map((pair) => pair[1]),
      distances: lod.map((pair) => pair[0]),
    };
  }
}

// Helper to create simplified geometry for LOD
function createSimplifiedGeometry(
  geometry: THREE.BufferGeometry,
  ratio: number,
): THREE.BufferGeometry {
  const targetCount =
    Math.floor((geometry.index!.array.length * ratio) / 3) * 3;
  const lodGeometry = geometry.clone();

  const dstIndexArray = MeshoptSimplifier.simplify(
    new Uint32Array(lodGeometry.index!.array),
    new Float32Array(lodGeometry.attributes.position.array),
    3,
    targetCount,
    0.02,
    ["LockBorder"],
  )[0];

  lodGeometry.index!.array.set(dstIndexArray);
  lodGeometry.index!.needsUpdate = true;
  lodGeometry.setDrawRange(0, dstIndexArray.length);

  return lodGeometry;
}

/**
 * Shared base component for batched mesh rendering using vanilla THREE.InstancedMesh
 * with multi-mesh LOD support and distance-based sorting for proper alpha compositing.
 */
export const BatchedMeshBase = React.forwardRef<
  THREE.Group,
  {
    // Data for instance positions and orientations
    batched_positions: Uint8Array;
    batched_wxyzs: Uint8Array;
    batched_scales: Uint8Array | null;
    batched_colors: Uint8Array | null;

    // Geometry info
    geometry: THREE.BufferGeometry;

    // Material info
    material: THREE.Material | THREE.Material[];

    // Rendering options
    lod: LodSetting;
    cast_shadow: boolean;
    receive_shadow: boolean;

    // Optional props
    clickable?: boolean;
  }
>(function BatchedMeshBase(props, ref) {
  const groupRef = useRef<THREE.Group>(null);
  const lodLevels = useRef<LODLevel[]>([]);
  const instanceCount = useRef<number>(0);
  const sortedIndices = useRef<Uint32Array | null>(null);
  const instanceDistances = useRef<Float32Array | null>(null);
  const lastCameraPosition = useRef<THREE.Vector3>(new THREE.Vector3());
  const tempMatrix = useMemo(() => new THREE.Matrix4(), []);
  const tempPosition = useMemo(() => new THREE.Vector3(), []);
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), []);
  const tempScale = useMemo(() => new THREE.Vector3(1, 1, 1), []);
  const tempColor = useMemo(() => new THREE.Color(), []);
  const camera = useThree((state) => state.camera);

  // Forward the ref to the group
  React.useImperativeHandle(ref, () => groupRef.current!, []);

  // Initialize LOD levels and meshes
  useEffect(() => {
    const count =
      props.batched_positions.byteLength / (3 * Float32Array.BYTES_PER_ELEMENT);
    instanceCount.current = count;

    // Clean up previous LOD levels
    lodLevels.current.forEach((level) => {
      level.mesh.dispose();
      level.geometry.dispose();
      if (Array.isArray(level.material)) {
        level.material.forEach((m) => m.dispose());
      } else {
        level.material.dispose();
      }
    });
    lodLevels.current = [];

    if (groupRef.current) {
      groupRef.current.clear();
    }

    // Create base mesh (LOD 0)
    const baseMesh = new THREE.InstancedMesh(
      props.geometry,
      props.material,
      count
    );
    baseMesh.castShadow = props.cast_shadow;
    baseMesh.receiveShadow = props.receive_shadow;

    if (groupRef.current) {
      groupRef.current.add(baseMesh);
    }

    const newLevels: LODLevel[] = [
      {
        mesh: baseMesh,
        geometry: props.geometry,
        material: props.material,
        distance: 0,
        ratio: 1.0,
      },
    ];

    // Create additional LOD levels if needed
    const { ratios, distances } = computeLODSettings(props.geometry, props.lod);

    ratios.forEach((ratio, index) => {
      const lodGeometry = createSimplifiedGeometry(props.geometry, ratio);
      const lodMaterial = Array.isArray(props.material)
        ? props.material.map((m) => m.clone())
        : props.material.clone();

      const lodMesh = new THREE.InstancedMesh(
        lodGeometry,
        lodMaterial,
        count
      );
      lodMesh.castShadow = props.cast_shadow;
      lodMesh.receiveShadow = props.receive_shadow;

      if (groupRef.current) {
        groupRef.current.add(lodMesh);
      }

      newLevels.push({
        mesh: lodMesh,
        geometry: lodGeometry,
        material: lodMaterial,
        distance: distances[index],
        ratio,
      });
    });

    lodLevels.current = newLevels;

    // Initialize sorting arrays
    sortedIndices.current = new Uint32Array(count);
    instanceDistances.current = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      sortedIndices.current[i] = i;
    }

    return () => {
      // Cleanup will happen in next effect run
    };
  }, [props.geometry, props.material, props.lod, props.batched_positions.byteLength]);

  // Update instance transforms
  useEffect(() => {
    const count =
      props.batched_positions.byteLength / (3 * Float32Array.BYTES_PER_ELEMENT);

    if (count !== instanceCount.current) {
      return; // Will be handled by the initialization effect
    }

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

    // Update transforms for all instances in all LOD levels
    for (let i = 0; i < count; i++) {
      const posOffset = i * 3 * 4;
      const wxyzOffset = i * 4 * 4;
      const scaleOffset =
        props.batched_scales &&
        props.batched_scales.byteLength ===
          (props.batched_wxyzs.byteLength / 4) * 3
          ? i * 3 * 4
          : i * 4;

      tempPosition.set(
        positionsView.getFloat32(posOffset, true),
        positionsView.getFloat32(posOffset + 4, true),
        positionsView.getFloat32(posOffset + 8, true),
      );

      tempQuaternion.set(
        wxyzsView.getFloat32(wxyzOffset + 4, true),
        wxyzsView.getFloat32(wxyzOffset + 8, true),
        wxyzsView.getFloat32(wxyzOffset + 12, true),
        wxyzsView.getFloat32(wxyzOffset, true),
      );

      if (scalesView) {
        if (
          props.batched_scales!.byteLength ===
          (props.batched_wxyzs.byteLength / 4) * 3
        ) {
          tempScale.set(
            scalesView.getFloat32(scaleOffset, true),
            scalesView.getFloat32(scaleOffset + 4, true),
            scalesView.getFloat32(scaleOffset + 8, true),
          );
        } else {
          const scale = scalesView.getFloat32(scaleOffset, true);
          tempScale.setScalar(scale);
        }
      } else {
        tempScale.set(1, 1, 1);
      }

      tempMatrix.compose(tempPosition, tempQuaternion, tempScale);

      // Apply to all LOD meshes
      lodLevels.current.forEach((level) => {
        level.mesh.setMatrixAt(i, tempMatrix);
      });
    }

    // Mark all instance matrices as needing update
    lodLevels.current.forEach((level) => {
      if (level.mesh.instanceMatrix) {
        level.mesh.instanceMatrix.needsUpdate = true;
      }
    });
  }, [
    props.batched_positions,
    props.batched_wxyzs,
    props.batched_scales,
  ]);

  // Update instance colors
  useEffect(() => {
    if (!props.batched_colors) return;

    const count = instanceCount.current;
    const colors = props.batched_colors;

    lodLevels.current.forEach((level) => {
      for (let i = 0; i < count; i++) {
        if (colors.byteLength === 3) {
          tempColor.setRGB(
            colors[0] / 255,
            colors[1] / 255,
            colors[2] / 255,
          );
        } else if (colors.byteLength === count * 3) {
          tempColor.setRGB(
            colors[i * 3] / 255,
            colors[i * 3 + 1] / 255,
            colors[i * 3 + 2] / 255,
          );
        } else {
          console.error(
            `Invalid batched_colors length: ${
              colors.byteLength
            }, expected 3 or ${count * 3}`,
          );
          tempColor.setRGB(1, 1, 1);
        }
        level.mesh.setColorAt(i, tempColor);
      }

      if (level.mesh.instanceColor) {
        level.mesh.instanceColor.needsUpdate = true;
      }
    });
  }, [props.batched_colors]);

  // Update shadow settings
  useEffect(() => {
    lodLevels.current.forEach((level) => {
      level.mesh.castShadow = props.cast_shadow;
      level.mesh.receiveShadow = props.receive_shadow;
    });
  }, [props.cast_shadow, props.receive_shadow]);

  // Sort instances and update LOD visibility
  useFrame(() => {
    if (!groupRef.current || lodLevels.current.length === 0) return;

    const count = instanceCount.current;
    if (count === 0) return;

    // Check if camera has moved significantly (threshold: 0.1 units)
    const cameraMovedSignificantly =
      lastCameraPosition.current.distanceToSquared(camera.position) > 0.01;

    if (!cameraMovedSignificantly && sortedIndices.current) {
      // Only update LOD visibility without re-sorting
      updateLODVisibility();
      return;
    }

    // Update last camera position
    lastCameraPosition.current.copy(camera.position);

    // Calculate distances for all instances
    const distances = instanceDistances.current!;
    let minDepth = Infinity;
    let maxDepth = -Infinity;

    for (let i = 0; i < count; i++) {
      // Get world position of instance
      lodLevels.current[0].mesh.getMatrixAt(i, tempMatrix);
      tempPosition.setFromMatrixPosition(tempMatrix);

      const distance = tempPosition.distanceTo(camera.position);
      distances[i] = distance;
      minDepth = Math.min(minDepth, distance);
      maxDepth = Math.max(maxDepth, distance);
    }

    // Perform counting sort (16-bit resolution)
    const depthRange = maxDepth - minDepth;
    if (depthRange < 0.001) {
      // All instances at same distance, no need to sort
      updateLODVisibility();
      return;
    }

    const depthInv = (256 * 256 - 1) / depthRange;
    const counts = new Uint32Array(256 * 256);
    const quantized = new Uint32Array(count);

    // Quantize and count
    for (let i = 0; i < count; i++) {
      quantized[i] = ((distances[i] - minDepth) * depthInv) | 0;
      counts[quantized[i]]++;
    }

    // Compute starting positions
    const starts = new Uint32Array(256 * 256);
    for (let i = 1; i < 256 * 256; i++) {
      starts[i] = starts[i - 1] + counts[i - 1];
    }

    // Build sorted index array
    const sorted = sortedIndices.current!;
    for (let i = 0; i < count; i++) {
      sorted[starts[quantized[i]]++] = i;
    }

    // Update LOD visibility based on sorted order
    updateLODVisibility();
  });

  function updateLODVisibility() {
    const count = instanceCount.current;
    const sorted = sortedIndices.current!;
    const distances = instanceDistances.current!;

    if (lodLevels.current.length <= 1) {
      // No LOD, just update render order for alpha compositing
      const baseMesh = lodLevels.current[0].mesh;

      // Rearrange instances based on sorted order
      const tempMatrices = new Float32Array(count * 16);
      const tempColors = baseMesh.instanceColor ? new Float32Array(count * 3) : null;

      // Save current state
      for (let i = 0; i < count; i++) {
        baseMesh.getMatrixAt(i, tempMatrix);
        tempMatrix.toArray(tempMatrices, i * 16);

        if (tempColors && baseMesh.instanceColor) {
          baseMesh.getColorAt(i, tempColor);
          tempColor.toArray(tempColors, i * 3);
        }
      }

      // Apply sorted order (front to back for alpha)
      for (let i = 0; i < count; i++) {
        const srcIdx = sorted[i];
        tempMatrix.fromArray(tempMatrices, srcIdx * 16);
        baseMesh.setMatrixAt(i, tempMatrix);

        if (tempColors && baseMesh.instanceColor) {
          tempColor.fromArray(tempColors, srcIdx * 3);
          baseMesh.setColorAt(i, tempColor);
        }
      }

      if (baseMesh.instanceMatrix) {
        baseMesh.instanceMatrix.needsUpdate = true;
      }
      if (baseMesh.instanceColor) {
        baseMesh.instanceColor.needsUpdate = true;
      }

      // Update count to render all
      baseMesh.count = count;
      return;
    }

    // Multi-LOD case: organize instances by LOD level
    const lodCounts = new Array(lodLevels.current.length).fill(0);
    const lodInstances: number[][] = lodLevels.current.map(() => []);

    // Assign each instance to appropriate LOD based on distance
    for (let i = 0; i < count; i++) {
      const instanceIdx = sorted[i];
      const distance = distances[instanceIdx];

      // Find appropriate LOD level
      let lodIndex = 0;
      for (let l = lodLevels.current.length - 1; l >= 0; l--) {
        if (distance >= lodLevels.current[l].distance) {
          lodIndex = l;
          break;
        }
      }

      lodInstances[lodIndex].push(instanceIdx);
      lodCounts[lodIndex]++;
    }

    // Update each LOD mesh with its instances
    lodLevels.current.forEach((level, lodIndex) => {
      const instances = lodInstances[lodIndex];
      const lodCount = instances.length;

      if (lodCount === 0) {
        // No instances at this LOD level
        level.mesh.count = 0;
        return;
      }

      // Rearrange instances for this LOD (sorted by distance)
      for (let i = 0; i < lodCount; i++) {
        const srcIdx = instances[i];

        // Copy transform from source instance
        lodLevels.current[0].mesh.getMatrixAt(srcIdx, tempMatrix);
        level.mesh.setMatrixAt(i, tempMatrix);

        // Copy color if available
        if (level.mesh.instanceColor && lodLevels.current[0].mesh.instanceColor) {
          lodLevels.current[0].mesh.getColorAt(srcIdx, tempColor);
          level.mesh.setColorAt(i, tempColor);
        }
      }

      // Hide remaining instances by placing them far away
      tempMatrix.makeScale(0, 0, 0);
      tempMatrix.setPosition(0, -100000, 0);
      for (let i = lodCount; i < count; i++) {
        level.mesh.setMatrixAt(i, tempMatrix);
      }

      if (level.mesh.instanceMatrix) {
        level.mesh.instanceMatrix.needsUpdate = true;
      }
      if (level.mesh.instanceColor) {
        level.mesh.instanceColor.needsUpdate = true;
      }

      // Update render count
      level.mesh.count = lodCount;
    });
  }

  return (
    <>
      <group ref={groupRef} />
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