import * as THREE from "three";
import React from "react";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { disposeMaterial } from "./MeshUtils";
import { GLTF, GLTFLoader, DRACOLoader } from "three-stdlib";

// We use a CDN for Draco. We could move this locally if we want to use Viser offline.
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath("https://www.gstatic.com/draco/v1/decoders/");

/**
 * Dispose a 3D object and its resources
 */
export function disposeNode(node: any) {
  if (node instanceof THREE.Mesh) {
    if (node.geometry) {
      node.geometry.dispose();
    }
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach((material) => {
          disposeMaterial(material);
        });
      } else {
        disposeMaterial(node.material);
      }
    }
  }
}

/**
 * Custom hook for loading a GLB model
 */
export function useGlbLoader(glb_data: Uint8Array, smoothShading: boolean = true) {
  // State for loaded model and meshes
  const [gltf, setGltf] = React.useState<GLTF>();
  const [meshes, setMeshes] = React.useState<THREE.Mesh[]>([]);

  // Animation mixer reference
  const mixerRef = React.useRef<THREE.AnimationMixer | null>(null);

  // Load the GLB model
  React.useEffect(() => {
    const loader = new GLTFLoader();
    loader.setDRACOLoader(dracoLoader);
    loader.parse(
      new Uint8Array(glb_data).buffer,
      "",
      (gltf) => {
        // Setup animations if present
        if (gltf.animations && gltf.animations.length) {
          mixerRef.current = new THREE.AnimationMixer(gltf.scene);
          gltf.animations.forEach((clip) => {
            mixerRef.current!.clipAction(clip).play();
          });
        }

        // Process all meshes in the scene
        const meshes: THREE.Mesh[] = [];
        gltf?.scene.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            const vertsBefore = obj.geometry.attributes.position?.count || 0;
            const normalsBefore = obj.geometry.attributes.normal?.array.slice(0, 15);
            
            obj.geometry = BufferGeometryUtils.mergeVertices(obj.geometry);
            
            const vertsAfterMerge = obj.geometry.attributes.position?.count || 0;
            
            // Delete normals to force recomputation
            obj.geometry.deleteAttribute('normal');
            obj.geometry.computeVertexNormals();
            
            const normalsAfter = obj.geometry.attributes.normal?.array.slice(0, 15);
            
            // Set smooth or flat shading on materials based on argument
            if (obj.material) {
              const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
              materials.forEach((mat) => {
                if (mat instanceof THREE.MeshStandardMaterial || 
                    mat instanceof THREE.MeshPhongMaterial ||
                    mat instanceof THREE.MeshLambertMaterial) {
                  mat.flatShading = !smoothShading;
                  mat.needsUpdate = true;
                }
              });
            }
            
            console.log(`[URDF Debug] Mesh: ${obj.name}`);
            console.log(`  Vertices: ${vertsBefore} -> ${vertsAfterMerge} (merged ${vertsBefore - vertsAfterMerge})`);
            console.log(`  Normals before compute:`, normalsBefore);
            console.log(`  Normals after compute:`, normalsAfter);
            
            obj.geometry.computeBoundingSphere();
            meshes.push(obj);
          }
        });

        setMeshes(meshes);
        setGltf(gltf);
      },
      (error) => {
        console.log("Error loading GLB!");
        console.log(error);
      },
    );

    // Cleanup function
    return () => {
      if (mixerRef.current) mixerRef.current.stopAllAction();

      // Attempt to free resources
      if (gltf) {
        gltf.scene.traverse(disposeNode);
      }
    };
  }, [glb_data]);

  // Return the loaded model, meshes, and mixer for animation updates
  return { gltf, meshes, mixerRef };
}
