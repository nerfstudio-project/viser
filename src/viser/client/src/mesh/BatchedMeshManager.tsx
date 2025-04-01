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
    scale?: number,
    castShadow: boolean = true,
    receiveShadow: boolean = true,
  ) {
    this.geometry = geometry.clone();
    this.instancedMesh = new InstancedMesh2(this.geometry, material);
    this.instancedMesh.castShadow = castShadow;
    this.instancedMesh.receiveShadow = receiveShadow;

    // Setup LODs if needed
    if (lodSetting !== "off") {
      this.setupLODs(this.geometry, material, lodSetting, castShadow, receiveShadow, scale);
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

  private addLODs(mesh: THREE.Mesh, ratios: number[], distances: number[]) {
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
      
      // Ensure we use a cloned material to avoid shared material issues
      let lodMaterial: THREE.Material;
      if (mesh.material instanceof THREE.MeshToonMaterial) {
        // For toon materials, create a new one with the same properties
        const toonMat = mesh.material as THREE.MeshToonMaterial;
        // Determine if it's toon3 or toon5 based on the gradient map size
        const shades = toonMat.gradientMap && toonMat.gradientMap.image && 
                       toonMat.gradientMap.image.width === 5 ? 5 : 3;
                       
        // Create a new instance with the same properties
        lodMaterial = new THREE.MeshToonMaterial({
          gradientMap: toonMat.gradientMap,  // We'll use the same texture for now
          color: toonMat.color.clone(),
          wireframe: toonMat.wireframe,
          transparent: toonMat.transparent,
          opacity: toonMat.opacity,
          flatShading: toonMat.flatShading,
          side: toonMat.side,
        });
      } else {
        // Clone other material types
        lodMaterial = mesh.material.clone();
      }
      
      this.instancedMesh.addLOD(lodGeometry, lodMaterial, distances[index]);

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
      this.instancedMesh.LODinfo.objects.forEach(obj => {
        obj.castShadow = castShadow;
        obj.receiveShadow = receiveShadow;
      });
    }
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
    // Convert RGB array to integer (using bit shift)
    const colorHex = (props.color[0] << 16) | (props.color[1] << 8) | props.color[2];
    
    // Cache for gradient maps to avoid creating too many textures
    const gradientMaps = {
      toon3: undefined as THREE.DataTexture | undefined,
      toon5: undefined as THREE.DataTexture | undefined
    };
    
    // Helper to generate or retrieve gradient map for toon materials
    const getGradientMap = (shades: 3 | 5): THREE.DataTexture => {
      const key = shades === 3 ? 'toon3' : 'toon5';
      
      // Reuse existing texture if available
      if (gradientMaps[key]) {
        return gradientMaps[key]!;
      }
      
      // Create a new texture if needed
      const texture = new THREE.DataTexture(
        Uint8Array.from(shades === 3 ? [0, 128, 255] : [0, 64, 128, 192, 255]),
        shades,
        1,
        THREE.RedFormat,
      );
      texture.needsUpdate = true;
      
      // Cache for future use
      gradientMaps[key] = texture;
      return texture;
    };
    
    // Update materials with the provided properties
    this.forEachMaterial((material) => {
      // This function will replace a material on an object (main mesh or LOD)
      const replaceMaterialOnObject = (obj: THREE.Object3D, oldMaterial: THREE.Material, newMaterial: THREE.Material) => {
        if (obj instanceof THREE.Mesh) {
          if (obj.material === oldMaterial) {
            obj.material = newMaterial;
          }
        }
      };
      
      // Handle material type conversion if needed
      if (props.materialType === 'standard' && material instanceof THREE.MeshToonMaterial) {
        // Convert toon to standard material
        const stdMaterial = new THREE.MeshStandardMaterial();
        this.copyBasicMaterialProperties(material, stdMaterial);
        
        // Replace on main mesh if needed
        if (this.instancedMesh.material === material) {
          this.instancedMesh.material = stdMaterial;
        }
        
        // Replace on LOD meshes too
        if (this.instancedMesh.LODinfo?.objects) {
          this.instancedMesh.LODinfo.objects.forEach(obj => {
            replaceMaterialOnObject(obj, material, stdMaterial);
          });
        }
        
        // Dispose the old material
        material.dispose();
        
        // Use the new material for the rest of the updates
        material = stdMaterial;
      } 
      else if ((props.materialType === 'toon3' || props.materialType === 'toon5') && 
          !(material instanceof THREE.MeshToonMaterial)) {
        // Convert standard to toon material
        const toonMaterial = new THREE.MeshToonMaterial({
          gradientMap: getGradientMap(props.materialType === 'toon3' ? 3 : 5)
        });
        this.copyBasicMaterialProperties(material, toonMaterial);
        
        // Replace on main mesh if needed
        if (this.instancedMesh.material === material) {
          this.instancedMesh.material = toonMaterial;
        }
        
        // Replace on LOD meshes too
        if (this.instancedMesh.LODinfo?.objects) {
          this.instancedMesh.LODinfo.objects.forEach(obj => {
            replaceMaterialOnObject(obj, material, toonMaterial);
          });
        }
        
        // Dispose the old material
        material.dispose();
        
        // Use the new material for the rest of the updates
        material = toonMaterial;
      }
      
      // Update color - all material types have this
      if ('color' in material && material.color instanceof THREE.Color) {
        material.color.setHex(colorHex);
      }
      
      // Update wireframe - all material types have this
      if ('wireframe' in material) {
        material.wireframe = props.wireframe;
      }
      
      // Update opacity - all material types have this
      if ('opacity' in material) {
        if (props.opacity !== null) {
          material.opacity = props.opacity;
          if ('transparent' in material) {
            material.transparent = props.opacity < 1;
          }
        } else {
          material.opacity = 1.0;
          if ('transparent' in material) {
            material.transparent = false;
          }
        }
      }
      
      // Update flatShading - needs needsUpdate flag to take effect
      if ('flatShading' in material) {
        material.flatShading = props.flatShading;
        material.needsUpdate = true;
      }
      
      // Update side
      if ('side' in material) {
        material.side = props.side;
        material.needsUpdate = true;
      }
      
      // Update transparent flag
      if ('transparent' in material) {
        // If opacity is null, transparent is false, otherwise check if opacity < 1
        material.transparent = props.opacity !== null ? props.opacity < 1 : props.transparent;
      }
      
      // Update gradient map for toon materials
      if (material instanceof THREE.MeshToonMaterial) {
        if (props.materialType === 'toon3') {
          // Use our cached gradient map for better performance
          material.gradientMap = getGradientMap(3);
          material.needsUpdate = true;
        } else if (props.materialType === 'toon5') {
          // Use our cached gradient map for better performance
          material.gradientMap = getGradientMap(5);
          material.needsUpdate = true;
        }
      }
    });
  }
  
  /**
   * Backward compatible method for color updates
   * @param color RGB color values as a tuple [r, g, b]
   */
  updateMaterialColor(color: [number, number, number]) {
    // For backward compatibility, we need to include all required parameters
    // We'll use sensible defaults for the other required parameters
    this.updateMaterialProperties({
      color: color,
      wireframe: false,
      opacity: null,
      flatShading: false,
      side: THREE.FrontSide,
      transparent: false,
      materialType: "standard"
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
        this.instancedMesh.material.forEach(mat => {
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
            obj.material.forEach(mat => {
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
   * Helper to copy basic material properties from one material to another
   * @param source Source material to copy properties from
   * @param target Target material to copy properties to
   */
  private copyBasicMaterialProperties(source: THREE.Material, target: THREE.Material) {
    // Copy common properties that both materials might have
    if ('color' in source && 'color' in target) {
      (target as any).color.copy((source as any).color);
    }
    
    if ('wireframe' in source && 'wireframe' in target) {
      (target as any).wireframe = (source as any).wireframe;
    }
    
    if ('opacity' in source && 'opacity' in target) {
      (target as any).opacity = (source as any).opacity;
    }
    
    if ('transparent' in source && 'transparent' in target) {
      (target as any).transparent = (source as any).transparent;
    }
    
    if ('flatShading' in source && 'flatShading' in target) {
      (target as any).flatShading = (source as any).flatShading;
      target.needsUpdate = true;
    }
    
    if ('side' in source && 'side' in target) {
      (target as any).side = (source as any).side;
    }
    
    // Copy material-specific properties if they exist
    if ('roughness' in source && 'roughness' in target) {
      (target as any).roughness = (source as any).roughness;
    }
    
    if ('metalness' in source && 'metalness' in target) {
      (target as any).metalness = (source as any).metalness;
    }
    
    target.needsUpdate = true;
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
