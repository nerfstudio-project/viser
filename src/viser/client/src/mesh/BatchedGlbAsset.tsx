import React, { useMemo } from "react";
import * as THREE from "three";
import { BatchedGlbMessage } from "../WebsocketMessages";
import { useGlbLoader } from "./GlbLoaderUtils";
import { ViewerContext } from "../ViewerContext";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { BatchedMeshBase } from "./BatchedMeshBase";

/**
 * Component for rendering batched/instanced GLB models
 *
 * Note: Batched GLB has some limitations:
 * - Animations are not supported
 * - The hierarchy in the GLB is flattened
 * - Each mesh in the GLB is instanced separately
 */
export const BatchedGlbAsset = React.forwardRef<THREE.Group, BatchedGlbMessage>(
  function BatchedGlbAsset(message, ref) {
    const viewer = React.useContext(ViewerContext)!;
    const clickable =
      viewer.useSceneTree(
        (state) => state.nodeFromName[message.name]?.clickable,
      ) ?? false;

    // Note: We don't support animations for batched meshes.
    const { gltf } = useGlbLoader(message.props.glb_data);

    // Extract geometry and materials from the GLB.
    const { geometry, material } = useMemo(() => {
      if (!gltf) return { geometry: null, material: null };

      // Collect meshes and their transforms from the original scene.
      const geometries: THREE.BufferGeometry[] = [];
      const materials: THREE.Material[] = [];
      gltf.scene.traverse((node) => {
        if (node instanceof THREE.Mesh && node.parent) {
          // Apply any transforms from the model hierarchy to the geometry.
          (node.geometry as THREE.BufferGeometry).applyMatrix4(
            node.matrixWorld,
          );
          geometries.push(node.geometry);
          materials.push(node.material);
        }
      });

      // Merge geometries if needed.
      const mergedGeometry =
        geometries.length === 1
          ? geometries[0].clone()
          : mergeGeometries(geometries, true);

      // Use either a single material or an array.
      const finalMaterial = materials.length === 1 ? materials[0] : materials;

      return {
        geometry: mergedGeometry,
        material: finalMaterial,
      };
    }, [gltf]);

    if (!geometry || !material) return null;

    return (
      <group ref={ref}>
        <BatchedMeshBase
          geometry={geometry}
          material={material}
          batched_positions={message.props.batched_positions}
          batched_wxyzs={message.props.batched_wxyzs}
          batched_scales={message.props.batched_scales}
          lod={message.props.lod}
          cast_shadow={message.props.cast_shadow}
          receive_shadow={message.props.receive_shadow}
          scale={message.props.scale}
          clickable={clickable}
        />
      </group>
    );
  },
);
