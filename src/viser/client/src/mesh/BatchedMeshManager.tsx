import * as THREE from "three";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { MeshoptSimplifier } from "meshoptimizer";
import { rgbToInt } from "./MeshUtils";

function getAutoLodSettings(
  mesh: THREE.Mesh,
  scale: number = 1,
): { ratios: number[]; distances: number[] } {
  // Heuristics for automatic LOD parameters.
  mesh.geometry.computeBoundingSphere();
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
    material: THREE.Material | THREE.Material[],
    lodSetting: "off" | "auto" | [number, number][],
    initialCount: number,
    scale?: number,
    castShadow: boolean = true,
    receiveShadow: boolean = true,
  ) {
    this.geometry = geometry;
    this.instancedMesh = new InstancedMesh2(this.geometry, material, {
      capacity: initialCount,
    });
    this.instancedMesh.resizeBuffers(initialCount);
    this.instancedMesh.castShadow = castShadow;
    this.instancedMesh.receiveShadow = receiveShadow;

    // Setup LODs if needed
    if (lodSetting !== "off") {
      this.setupLODs(
        this.geometry,
        material,
        lodSetting,
        castShadow,
        receiveShadow,
        scale,
      );
    }
  }

  private setupLODs(
    geometry: THREE.BufferGeometry,
    material: THREE.Material | THREE.Material[],
    lodSetting: "auto" | [number, number][],
    castShadow: boolean,
    receiveShadow: boolean,
    scale?: number,
  ) {
    const dummyMesh = new THREE.Mesh(geometry, material);

    if (lodSetting === "auto") {
      const { ratios, distances } = getAutoLodSettings(dummyMesh, scale);
      this.addLODs(dummyMesh, ratios, distances);
    } else {
      this.addLODs(
        dummyMesh,
        lodSetting.map((pair) => pair[1]),
        lodSetting.map((pair) => pair[0]),
      );
    }

    // Apply shadow settings to all LOD objects
    if (this.instancedMesh.LODinfo && this.instancedMesh.LODinfo.objects) {
      this.instancedMesh.LODinfo.objects.forEach((obj) => {
        obj.castShadow = castShadow;
        obj.receiveShadow = receiveShadow;
      });
    }
  }

  /**
   * Creates lower level of detail (LOD) versions of the mesh for efficient rendering at different distances
   * @param mesh Base mesh to create LODs from
   * @param ratios Array of ratios of vertices to keep for each LOD level
   * @param distances Array of distances at which to switch to each LOD level
   */
  private addLODs(mesh: THREE.Mesh, ratios: number[], distances: number[]) {
    ratios.forEach((ratio, index) => {
      // Calculate target triangle count based on the ratio
      const targetCount =
        Math.floor((mesh.geometry.index!.array.length * ratio) / 3) * 3;
      const lodGeometry = mesh.geometry.clone();

      // Use meshopt to simplify the geometry
      const dstIndexArray = MeshoptSimplifier.simplify(
        new Uint32Array(lodGeometry.index!.array),
        new Float32Array(lodGeometry.attributes.position.array),
        3,
        targetCount,
        0.01, // Error tolerance
        ["LockBorder"], // Prevents triangle flipping artifacts
      )[0];

      // Update the geometry with the simplified version
      lodGeometry.index!.array.set(dstIndexArray);
      lodGeometry.index!.needsUpdate = true;
      lodGeometry.setDrawRange(0, dstIndexArray.length);

      // Create a cloned material for this LOD level
      const lodMaterial = (
        Array.isArray(mesh.material) ? mesh.material[0] : mesh.material
      ).clone();

      // Add this LOD level to the instanced mesh
      this.instancedMesh.addLOD(lodGeometry, lodMaterial, distances[index]);

      // Store the geometry for proper disposal later
      this.lodGeometries.push(lodGeometry);
    });
  }

  // Reusable objects for transform calculations
  private readonly _tempPosition = new THREE.Vector3();
  private readonly _tempQuaternion = new THREE.Quaternion();
  private readonly _tempScale = new THREE.Vector3(1, 1, 1);
  private readonly _instanceWorldMatrix = new THREE.Matrix4();
  private readonly _meshMatrix = new THREE.Matrix4();

  /**
   * Update instance transforms without allocating new objects
   * @param batched_positions Raw bytes containing float32 position values (xyz)
   * @param batched_wxyzs Raw bytes containing float32 quaternion values (wxyz)
   * @param meshTransform Optional transform to apply to each instance
   */
  updateInstances(
    /** Raw bytes containing float32 position values (xyz) */
    batched_positions: Uint8Array,
    /** Raw bytes containing float32 quaternion values (wxyz) */
    batched_wxyzs: Uint8Array,
    meshTransform?: {
      position: THREE.Vector3;
      rotation: THREE.Quaternion;
      scale: THREE.Vector3;
    },
  ) {
    // Cache member variables to avoid 'this' lookup in loop.
    const tempPosition = this._tempPosition;
    const tempQuaternion = this._tempQuaternion;
    const tempScale = this._tempScale;
    const instanceWorldMatrix = this._instanceWorldMatrix;
    const meshMatrix = this._meshMatrix;

    // Pre-compute mesh matrix if needed (only once outside the loop).
    if (meshTransform) {
      meshMatrix.compose(
        meshTransform.position,
        meshTransform.rotation,
        tempScale, // Reuse scale vector (1,1,1)
      );
    }

    // Create DataViews to read float values directly from the Uint8Array.
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

    this.instancedMesh.updateInstances((obj, index) => {
      // Calculate byte offsets for reading float values.
      const posOffset = index * 3 * 4; // 3 floats, 4 bytes per float
      const wxyzOffset = index * 4 * 4; // 4 floats, 4 bytes per float

      // Read float values from DataView.
      tempPosition.set(
        positionsView.getFloat32(posOffset, true), // x
        positionsView.getFloat32(posOffset + 4, true), // y
        positionsView.getFloat32(posOffset + 8, true), // z
      );

      tempQuaternion.set(
        wxyzsView.getFloat32(wxyzOffset + 4, true), // x
        wxyzsView.getFloat32(wxyzOffset + 8, true), // y
        wxyzsView.getFloat32(wxyzOffset + 12, true), // z
        wxyzsView.getFloat32(wxyzOffset, true), // w (first value)
      );

      // Create instance world transform.
      instanceWorldMatrix.compose(tempPosition, tempQuaternion, tempScale);

      if (meshTransform) {
        // Combine transforms and apply (reuses the instanceWorldMatrix).
        instanceWorldMatrix.multiply(meshMatrix);
        obj.position.setFromMatrixPosition(instanceWorldMatrix);
        obj.quaternion.setFromRotationMatrix(instanceWorldMatrix);
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
    this.instancedMesh.computeBVH();
  }

  /** Get the instanced mesh object */
  getMesh() {
    return this.instancedMesh;
  }

  /**
   * Update shadow settings without recreating the mesh
   * @param castShadow Whether meshes should cast shadows
   * @param receiveShadow Whether meshes should receive shadows
   */
  updateShadowSettings(castShadow: boolean, receiveShadow: boolean) {
    // Update main instance mesh
    this.instancedMesh.castShadow = castShadow;
    this.instancedMesh.receiveShadow = receiveShadow;

    // Update all LOD objects
    if (this.instancedMesh.LODinfo?.objects) {
      this.instancedMesh.LODinfo.objects.forEach((obj) => {
        obj.castShadow = castShadow;
        obj.receiveShadow = receiveShadow;
      });
    }
  }

  // Static texture cache to prevent creating new textures on every update
  private static readonly gradientMapCache: {
    toon3?: THREE.DataTexture;
    toon5?: THREE.DataTexture;
  } = {};

  /**
   * Creates or retrieves a gradient map texture for toon materials
   * @param shades Number of shades (3 or 5)
   * @returns A DataTexture suitable for use as a gradient map
   */
  private getGradientMap(shades: 3 | 5): THREE.DataTexture {
    const key = shades === 3 ? "toon3" : "toon5";

    // Reuse existing texture if available
    if (BatchedMeshManager.gradientMapCache[key]) {
      return BatchedMeshManager.gradientMapCache[key]!;
    }

    // Create a new texture if needed
    const data =
      shades === 3
        ? Uint8Array.from([0, 128, 255])
        : Uint8Array.from([0, 64, 128, 192, 255]);

    const texture = new THREE.DataTexture(data, shades, 1, THREE.RedFormat);
    texture.needsUpdate = true;

    // Cache for future use
    BatchedMeshManager.gradientMapCache[key] = texture;
    return texture;
  }

  /**
   * Update all material properties without recreating the mesh
   * @param props Object containing material properties to update
   */
  updateMaterialProperties(props: {
    color: [number, number, number];
    wireframe: boolean;
    opacity: number | null;
    flatShading: boolean;
    side: THREE.Side;
    transparent: boolean;
    materialType: "standard" | "toon3" | "toon5";
  }) {
    // Update materials with the provided properties
    this.forEachMaterial((material) => {
      const newMaterial =
        props.materialType === "standard"
          ? new THREE.MeshStandardMaterial()
          : new THREE.MeshToonMaterial({
              gradientMap: this.getGradientMap(
                props.materialType === "toon3" ? 3 : 5,
              ),
            });
      // Replace material on all meshes
      this.replaceMaterialOnAllMeshes(material, newMaterial);
      material.dispose();

      newMaterial.color.setHex(rgbToInt(props.color));
      newMaterial.wireframe = props.wireframe;
      newMaterial.opacity = props.opacity !== null ? props.opacity : 1.0;
      if (newMaterial instanceof THREE.MeshStandardMaterial) {
        newMaterial.flatShading = props.flatShading;
      }
      newMaterial.transparent =
        props.opacity !== null ? props.opacity < 1 : props.transparent;
      newMaterial.side = props.side;
      newMaterial.needsUpdate = true;
    });
  }

  /**
   * Helper function to iterate over all materials in the instanced mesh and its LODs
   * @param callback Function to call for each unique material
   */
  private forEachMaterial(callback: (material: THREE.Material) => void) {
    // Track processed materials to avoid duplicates or repeated processing
    const processedMaterials = new Set<THREE.Material>();

    // Process the main material first
    if (this.instancedMesh.material) {
      if (Array.isArray(this.instancedMesh.material)) {
        this.instancedMesh.material.forEach((mat) => {
          if (!processedMaterials.has(mat)) {
            processedMaterials.add(mat);
            callback(mat);
          }
        });
      } else {
        processedMaterials.add(this.instancedMesh.material);
        callback(this.instancedMesh.material);
      }
    }

    // Then process materials from LOD levels
    if (this.instancedMesh.LODinfo?.objects) {
      this.instancedMesh.LODinfo.objects.forEach((obj) => {
        if (obj instanceof THREE.Mesh && obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((mat) => {
              if (!processedMaterials.has(mat)) {
                processedMaterials.add(mat);
                callback(mat);
              }
            });
          } else if (!processedMaterials.has(obj.material)) {
            processedMaterials.add(obj.material);
            callback(obj.material);
          }
        }
      });
    }
  }

  /**
   * Replaces material on all meshes (main mesh and LODs)
   * @param oldMaterial The material to replace
   * @param newMaterial The material to replace it with
   */
  private replaceMaterialOnAllMeshes(
    oldMaterial: THREE.Material,
    newMaterial: THREE.Material,
  ) {
    // Replace on main mesh if needed
    if (this.instancedMesh.material === oldMaterial) {
      this.instancedMesh.material = newMaterial;
    }

    // Replace on LOD meshes too
    if (this.instancedMesh.LODinfo?.objects) {
      this.instancedMesh.LODinfo.objects.forEach((obj) => {
        if (obj instanceof THREE.Mesh && obj.material === oldMaterial) {
          obj.material = newMaterial;
        }
      });
    }
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
