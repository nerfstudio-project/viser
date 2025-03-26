import * as THREE from "three";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { MeshoptSimplifier } from "meshoptimizer";

function getAutoLodSettings(
  mesh: THREE.Mesh,
  scale: number = 1,
): { ratios: number[]; distances: number[] } {
  // Heuristics for automatic LOD parameters.
  const geometry = mesh.geometry;
  const boundingRadius = geometry.boundingSphere!.radius * scale;
  const vertexCount = geometry.attributes.position.count;

  // 1. Compute LOD ratios, based on vertex count.
  let ratios: number[];
  if (vertexCount > 10_000) {
    ratios = [0.2, 0.05, 0.01]; // very complex
  } else if (vertexCount > 2_000) {
    ratios = [0.4, 0.1, 0.03]; // medium complex
  } else if (vertexCount > 500) {
    ratios = [0.6, 0.2, 0.05]; // light
  } else {
    ratios = [0.85, 0.4, 0.1]; // already simple
  }

  // 2. Compute LOD distances, based on bounding radius.
  const sizeFactor = Math.sqrt(boundingRadius + 1e-5);
  const baseMultipliers = [1, 2, 3]; // distance "steps" for LOD switching
  const distances = baseMultipliers.map((m) => m * sizeFactor);

  return { ratios, distances };
}

/**
 * Helper class to manage batched mesh instances and ensure proper resource disposal
 * This class ensures that all resources (geometries, LODs) are properly cleaned up
 */
export class BatchedMeshManager {
  private instancedMesh: InstancedMesh2;
  private lodGeometries: THREE.BufferGeometry[] = [];
  private geometry: THREE.BufferGeometry;

  constructor(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    numInstances: number,
    lodSetting: "off" | "auto" | [number, number][],
    castShadow: boolean,
    receiveShadow: boolean,
    scale?: number,
  ) {
    this.geometry = geometry.clone();
    this.instancedMesh = new InstancedMesh2(this.geometry, material);
    this.instancedMesh.castShadow = castShadow;
    this.instancedMesh.receiveShadow = receiveShadow;

    // Setup LODs if needed
    if (lodSetting !== "off") {
      this.setupLODs(this.geometry, material, lodSetting, castShadow, scale);
    }

    // Setup instances
    this.instancedMesh.addInstances(numInstances, () => {});
    this.instancedMesh.computeBVH();
  }

  private setupLODs(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    lodSetting: "auto" | [number, number][],
    castShadow: boolean,
    scale?: number,
  ) {
    const dummyMesh = new THREE.Mesh(geometry, material);

    if (lodSetting === "auto") {
      const { ratios, distances } = getAutoLodSettings(dummyMesh, scale);
      this.addLODs(dummyMesh, ratios, distances, castShadow);
    } else {
      this.addLODs(
        dummyMesh,
        lodSetting.map((pair) => pair[1]),
        lodSetting.map((pair) => pair[0]),
        castShadow,
      );
    }
  }

  private addLODs(
    mesh: THREE.Mesh,
    ratios: number[],
    distances: number[],
    castShadow: boolean,
  ) {
    ratios.forEach((ratio, index) => {
      const targetCount =
        Math.floor((mesh.geometry.index!.array.length * ratio) / 3) * 3;
      const lodGeometry = mesh.geometry.clone();

      const dstIndexArray = MeshoptSimplifier.simplify(
        new Uint32Array(lodGeometry.index!.array),
        new Float32Array(lodGeometry.attributes.position.array),
        3,
        targetCount,
        0.01, // Error tolerance.
        ["LockBorder"], // Important to avoid triangle flipping artifacts.
      )[0];

      lodGeometry.index!.array.set(dstIndexArray);
      lodGeometry.index!.needsUpdate = true;
      lodGeometry.setDrawRange(0, dstIndexArray.length);
      this.instancedMesh.addLOD(lodGeometry, mesh.material, distances[index]);

      if (castShadow) {
        this.instancedMesh.addShadowLOD(lodGeometry, distances[index]);
      }

      // Store the geometry for proper disposal later
      this.lodGeometries.push(lodGeometry);
    });
  }

  /** Update instance transforms */
  updateInstances(
    batched_positions: Float32Array,
    batched_wxyzs: Float32Array,
    meshTransform?: {
      position: THREE.Vector3;
      rotation: THREE.Quaternion;
      scale: THREE.Vector3;
    },
  ) {
    this.instancedMesh.updateInstances((obj, index) => {
      // Create instance world transform
      const instanceWorldMatrix = new THREE.Matrix4().compose(
        new THREE.Vector3(
          batched_positions[index * 3 + 0],
          batched_positions[index * 3 + 1],
          batched_positions[index * 3 + 2],
        ),
        new THREE.Quaternion(
          batched_wxyzs[index * 4 + 1],
          batched_wxyzs[index * 4 + 2],
          batched_wxyzs[index * 4 + 3],
          batched_wxyzs[index * 4 + 0],
        ),
        new THREE.Vector3(1, 1, 1),
      );

      if (meshTransform) {
        // Apply mesh's original transform relative to the instance
        const meshMatrix = new THREE.Matrix4().compose(
          meshTransform.position,
          meshTransform.rotation,
          new THREE.Vector3(1, 1, 1),
        );

        // Combine transforms and apply
        const finalMatrix = instanceWorldMatrix.multiply(meshMatrix);
        obj.position.setFromMatrixPosition(finalMatrix);
        obj.quaternion.setFromRotationMatrix(finalMatrix);
        obj.scale.copy(meshTransform.scale);
      } else {
        // Direct instance transform without mesh offset
        obj.position.setFromMatrixPosition(instanceWorldMatrix);
        obj.quaternion.setFromRotationMatrix(instanceWorldMatrix);
      }
    });
  }

  /** Update the number of instances */
  setInstanceCount(numInstances: number) {
    this.instancedMesh.clearInstances();
    this.instancedMesh.addInstances(numInstances, () => {});
  }

  /** Get the instanced mesh object */
  getMesh() {
    return this.instancedMesh;
  }

  /** Dispose all resources */
  dispose() {
    // Dispose all LOD geometries
    for (const geometry of this.lodGeometries) {
      geometry.dispose();
    }

    // The instancedMesh will dispose its main geometry and material
    this.instancedMesh.disposeBVH();
    this.instancedMesh.dispose();
    this.geometry.dispose();
  }
}
