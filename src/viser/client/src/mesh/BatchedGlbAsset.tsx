import React from "react";
import * as THREE from "three";
import { BatchedGlbMessage } from "../WebsocketMessages";
import { useGlbLoader } from "./GlbLoaderUtils";
import { BatchedMeshManager } from "./BatchedMeshManager";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";
import { ViewerContext } from "../ViewerContext";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils";

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
    // We don't pass shadow settings to the GLB loader - we'll apply them manually.
    const { gltf } = useGlbLoader(message.props.glb_data);

    // Use memoization to create mesh managers and transforms when GLB loads.
    const meshState = React.useMemo(() => {
      if (!gltf) return null;

      // Collect meshes and their transforms from the original scene.
      const geometries: THREE.BufferGeometry[] = [];
      const materials: THREE.Material[] = [];
      gltf.scene.traverse((node) => {
        if (node instanceof THREE.Mesh && node.parent) {
          (node.geometry as THREE.BufferGeometry).applyMatrix4(
            node.matrixWorld,
          );
          geometries.push(node.geometry);
          materials.push(node.material);
        }
      });

      // Create manager without shadow settings to avoid recreation - we'll set them in useEffect
      console.log(geometries.length, materials.length);
      const manager = new BatchedMeshManager(
        geometries.length == 1 ? geometries[0] : mergeGeometries(geometries),
        materials.length == 1 ? materials[0] : materials,
        message.props.lod,
        message.props.batched_positions.byteLength /
          (3 * Float32Array.BYTES_PER_ELEMENT),
      );

      return { manager };
    }, [gltf, message.props.lod]);

    // 0. Update instance count.
    React.useEffect(() => {
      if (meshState === null) return;
      // Create instanced mesh with LOD.
      const numInstances =
        message.props.batched_positions.byteLength /
        (3 * Float32Array.BYTES_PER_ELEMENT);
      meshState.manager.setInstanceCount(numInstances);
    }, [meshState, message.props.batched_positions.byteLength]);

    // 1. Update instance transforms (positions and orientations)
    React.useEffect(() => {
      if (meshState === null) return;
      meshState.manager.updateInstances(
        message.props.batched_positions,
        message.props.batched_wxyzs,
      );
    }, [
      meshState,
      message.props.batched_positions,
      message.props.batched_wxyzs,
    ]);

    // 2. Update shadow settings - separate effect for better performance
    React.useEffect(() => {
      if (meshState === null) return;
      meshState.manager.updateShadowSettings(
        message.props.cast_shadow,
        message.props.receive_shadow,
      );
    }, [meshState, message.props.cast_shadow, message.props.receive_shadow]);

    // Clean up resources when dependencies change or component unmounts.
    React.useEffect(() => {
      return () => {
        if (meshState === null) return;
        // Dispose all batch managers.
        meshState.manager.dispose();
      };
    }, [meshState]);

    if (!gltf || !meshState) return null;
    const mesh = meshState.manager.getMesh();
    return (
      <group ref={ref}>
        <primitive
          object={meshState.manager.getMesh()}
          scale={message.props.scale}
        />

        {/* Add outlines for each mesh in the GLB asset */}
        {clickable && mesh.geometry && (
          <BatchedMeshHoverOutlines
            geometry={mesh.geometry}
            batched_positions={
              message.props.batched_positions
            } /* Raw bytes containing float32 position values */
            batched_wxyzs={
              message.props.batched_wxyzs
            } /* Raw bytes containing float32 quaternion values */
          />
        )}
      </group>
    );
  },
);
