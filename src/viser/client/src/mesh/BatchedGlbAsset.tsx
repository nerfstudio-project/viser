import React from "react";
import * as THREE from "three";
import { BatchedGlbMessage } from "../WebsocketMessages";
import { useGlbLoader } from "./GlbLoaderUtils";
import { BatchedMeshManager } from "./BatchedMeshManager";
import { useFrame } from "@react-three/fiber";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";
import { ViewerContext } from "../ViewerContext";

/**
 * Component for rendering batched/instanced GLB models
 */
export const BatchedGlbAsset = React.forwardRef<THREE.Group, BatchedGlbMessage>(
  function BatchedGlbAsset(message, ref) {
    const viewer = React.useContext(ViewerContext)!;
    const clickable =
      viewer.useSceneTree(
        (state) => state.nodeFromName[message.name]?.clickable,
      ) ?? false;

    const { gltf, mixerRef } = useGlbLoader(
      message.props.glb_data,
      message.props.cast_shadow,
      message.props.receive_shadow,
    );

    // Update animations on each frame if mixer exists.
    useFrame((_, delta: number) => {
      mixerRef.current?.update(delta);
    });

    // Create Float32Arrays once for positions and orientations.
    const batched_positions = React.useMemo(
      () =>
        new Float32Array(
          message.props.batched_positions.buffer.slice(
            message.props.batched_positions.byteOffset,
            message.props.batched_positions.byteOffset +
              message.props.batched_positions.byteLength,
          ),
        ),
      [message.props.batched_positions],
    );

    const batched_wxyzs = React.useMemo(
      () =>
        new Float32Array(
          message.props.batched_wxyzs.buffer.slice(
            message.props.batched_wxyzs.byteOffset,
            message.props.batched_wxyzs.byteOffset +
              message.props.batched_wxyzs.byteLength,
          ),
        ),
      [message.props.batched_wxyzs],
    );

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
          const manager = new BatchedMeshManager(
            node.geometry,
            node.material,
            numInstances,
            message.props.lod,
            message.props.cast_shadow,
            message.props.receive_shadow,
            Math.max(scale.x, scale.y, scale.z),
          );

          // Add the instanced mesh to our group.
          managers.push(manager);
          instancedGroup.add(manager.getMesh());
        }
      });

      return { instancedGroup, managers, transforms };
    }, [
      gltf,
      message.props.lod,
      message.props.cast_shadow,
      message.props.receive_shadow,
      message.props.batched_positions.byteLength,
    ]);

    // Add new useEffect to handle position updates
    React.useEffect(() => {
      if (meshState && meshState.managers) {
        meshState.managers.forEach((manager, index) => {
          manager.updateInstances(
            batched_positions,
            batched_wxyzs,
            meshState.transforms[index]
          );
        });
      }
    }, [meshState, batched_positions, batched_wxyzs]);

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
                batched_positions={batched_positions}
                batched_wxyzs={batched_wxyzs}
                meshTransform={transform}
              />
            );
          })}
      </group>
    );
  },
);
