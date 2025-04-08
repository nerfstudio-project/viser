import React from "react";
import * as THREE from "three";
import { BatchedGlbMessage } from "../WebsocketMessages";
import { useGlbLoader } from "./GlbLoaderUtils";
import { BatchedMeshManager } from "./BatchedMeshManager";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";
import { ViewerContext } from "../ViewerContext";

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

      // Create a new group to hold our instanced meshes.
      const instancedGroup = new THREE.Group();
      const managers: BatchedMeshManager[] = [];
      const transforms: {
        position: THREE.Vector3;
        rotation: THREE.Quaternion;
        scale: THREE.Vector3;
      }[] = [];

      // Collect meshes and their transforms from the original scene.
      gltf.scene.traverse((node) => {
        if (node instanceof THREE.Mesh && node.parent) {
          // Store transform info.
          const position = new THREE.Vector3();
          const scale = new THREE.Vector3();
          const quat = new THREE.Quaternion();
          node.getWorldPosition(position);
          node.getWorldScale(scale);
          node.getWorldQuaternion(quat);

          const transform = {
            position: position.clone(),
            rotation: quat.clone(),
            scale: scale.clone(),
          };
          transforms.push(transform);

          // Create instanced mesh with LOD.
          const numInstances =
            message.props.batched_positions.byteLength /
            (3 * Float32Array.BYTES_PER_ELEMENT);

          // Create manager without shadow settings to avoid recreation - we'll set them in useEffect
          const manager = new BatchedMeshManager(
            node.geometry,
            node.material,
            numInstances,
            message.props.lod,
            Math.max(scale.x, scale.y, scale.z),
          );

          // Add the instanced mesh to our group.
          managers.push(manager);
          instancedGroup.add(manager.getMesh());
        }
      });

      return { instancedGroup, managers, transforms };
    }, [gltf, message.props.lod, message.props.batched_positions.byteLength]);

    // 1. Update instance transforms (positions and orientations)
    React.useEffect(() => {
      if (meshState && meshState.managers) {
        meshState.managers.forEach((manager, index) => {
          // Pass buffer views directly to avoid creating new arrays.
          manager.updateInstances(
            message.props.batched_positions,
            message.props.batched_wxyzs,
            meshState.transforms[index],
          );
        });
      }
    }, [
      meshState,
      message.props.batched_positions,
      message.props.batched_wxyzs,
    ]);

    // 2. Update shadow settings - separate effect for better performance
    React.useEffect(() => {
      if (meshState && meshState.managers) {
        meshState.managers.forEach((manager) => {
          manager.updateShadowSettings(
            message.props.cast_shadow,
            message.props.receive_shadow,
          );
        });
      }
    }, [meshState, message.props.cast_shadow, message.props.receive_shadow]);

    // Clean up resources when dependencies change or component unmounts.
    React.useEffect(() => {
      return () => {
        if (meshState) {
          // Dispose all batch managers.
          meshState.managers.forEach((manager) => {
            manager.dispose();
          });

          // Clear the instanced group.
          if (meshState.instancedGroup) {
            meshState.instancedGroup.clear();
          }
        }
      };
    }, [meshState]);

    if (!gltf || !meshState) return null;

    return (
      <group ref={ref}>
        <primitive
          object={meshState.instancedGroup}
          scale={message.props.scale}
        />

        {/* Add outlines for each mesh in the GLB asset */}
        {clickable &&
          meshState.transforms.map((transform, index) => {
            // Get the mesh's geometry from the manager.
            const manager = meshState.managers[index];
            if (!manager) return null;

            const mesh = manager.getMesh();
            if (!mesh || !mesh.geometry) return null;

            return (
              <BatchedMeshHoverOutlines
                key={index}
                geometry={mesh.geometry}
                batched_positions={
                  message.props.batched_positions
                } /* Raw bytes containing float32 position values */
                batched_wxyzs={
                  message.props.batched_wxyzs
                } /* Raw bytes containing float32 quaternion values */
                meshTransform={transform}
              />
            );
          })}
      </group>
    );
  },
);
