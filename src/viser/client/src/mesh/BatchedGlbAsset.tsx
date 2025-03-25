import React from "react";
import * as THREE from "three";
import { BatchedGlbMessage } from "../WebsocketMessages";
import { useGlbLoader } from "./GlbLoaderUtils";
import { BatchedMeshManager, setupBatchedMesh } from "./BatchedMeshManager";
import { useFrame } from "@react-three/fiber";
import { BatchedMeshHoverOutlines } from "./BatchedMeshHoverOutlines";

/**
 * Component for rendering batched/instanced GLB models
 */
export const BatchedGlbAsset = React.forwardRef<THREE.Group, BatchedGlbMessage>(
  function BatchedGlbAsset(message, ref) {
    const { gltf, mixerRef } = useGlbLoader(
      message.props.glb_data,
      message.props.cast_shadow,
      message.props.receive_shadow,
    );

    // Update animations on each frame
    useFrame((_: any, delta: number) => {
      if (mixerRef.current) {
        mixerRef.current.update(delta);
      }
    });

    // Store transforms for mesh to instance mapping
    const [transforms, setTransforms] = React.useState<
      {
        position: THREE.Vector3;
        rotation: THREE.Quaternion;
        scale: THREE.Vector3;
      }[]
    >([]);

    // Use state to store mesh managers and scene
    const [meshState, setMeshState] = React.useState<{
      gltfScene: THREE.Group;
      managers: BatchedMeshManager[];
    } | null>(null);

    // Initialize mesh managers when the GLB loads
    React.useEffect(() => {
      if (!gltf) return;

      // Clean up previous managers if they exist
      if (meshState) {
        meshState.managers.forEach((manager) => {
          manager.dispose();
        });
      }

      const scene = gltf.scene.clone();
      const managers: BatchedMeshManager[] = [];
      const transforms: {
        position: THREE.Vector3;
        rotation: THREE.Quaternion;
        scale: THREE.Vector3;
      }[] = [];

      scene.traverse((node) => {
        if (node instanceof THREE.Mesh && node.parent) {
          // Store transform info
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

          // Create instanced mesh with LOD
          const numInstances =
            message.props.batched_positions.byteLength /
            (3 * Float32Array.BYTES_PER_ELEMENT);
          const manager = setupBatchedMesh(
            node.geometry.clone(),
            node.material,
            numInstances,
            message.props.lod,
            message.props.cast_shadow,
            Math.max(scale.x, scale.y, scale.z),
          );

          // Hide the original node
          node.visible = false;

          // Add the instanced mesh to the scene
          managers.push(manager);
          scene.add(manager.getMesh());
        }
      });

      setTransforms(transforms);
      setMeshState({ gltfScene: scene, managers });

      // Clean up when component unmounts or dependencies change
      return () => {
        if (meshState) {
          meshState.managers.forEach((manager) => {
            manager.dispose();
          });
        }
      };
    }, [
      gltf,
      message.props.lod,
      message.props.cast_shadow,
      message.props.batched_positions.byteLength,
    ]);

    // Create Float32Arrays once for positions and orientations
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

    // Handle updates to instance positions/orientations
    React.useEffect(() => {
      if (!meshState) return;

      // Update instance count if needed
      const newNumInstances = batched_positions.length / 3;

      // Update all mesh managers - use the arrays we already created
      meshState.managers.forEach((manager, mesh_index) => {
        manager.setInstanceCount(newNumInstances);
        manager.updateInstances(
          batched_positions,
          batched_wxyzs,
          transforms[mesh_index],
        );
      });
    }, [meshState, transforms, batched_positions, batched_wxyzs]);

    if (!gltf || !meshState) return null;

    return (
      <group ref={ref}>
        <primitive object={meshState.gltfScene} scale={message.props.scale} />

        {/* Add outlines for each mesh in the GLB asset */}
        {transforms.map((transform, index) => {
          // Get the mesh's geometry from the manager
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
