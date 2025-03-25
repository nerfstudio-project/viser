import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { SkinnedMeshMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";
import { ViewerContext } from "../ViewerContext";
import { useFrame } from "@react-three/fiber";

/**
 * Component for rendering skinned meshes with animations
 */
export const SkinnedMesh = React.forwardRef<
  THREE.SkinnedMesh,
  SkinnedMeshMessage
>(function SkinnedMesh(message, ref) {
  const viewer = React.useContext(ViewerContext)!;

  // Create material using memoization
  const material = React.useMemo(() => {
    return createStandardMaterial(message.props);
  }, [
    message.props.material,
    message.props.color,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
  ]);

  // Reference to bones for animation updates
  const bonesRef = React.useRef<THREE.Bone[]>();

  // Create geometry and skeleton using memoization
  const { geometry, skeleton } = React.useMemo(() => {
    // Setup geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(
          message.props.vertices.buffer.slice(
            message.props.vertices.byteOffset,
            message.props.vertices.byteOffset +
              message.props.vertices.byteLength,
          ),
        ),
        3,
      ),
    );
    geometry.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(
          message.props.faces.buffer.slice(
            message.props.faces.byteOffset,
            message.props.faces.byteOffset + message.props.faces.byteLength,
          ),
        ),
        1,
      ),
    );
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();

    // Setup skinned mesh bones
    const bone_wxyzs = new Float32Array(
      message.props.bone_wxyzs.buffer.slice(
        message.props.bone_wxyzs.byteOffset,
        message.props.bone_wxyzs.byteOffset +
          message.props.bone_wxyzs.byteLength,
      ),
    );
    const bone_positions = new Float32Array(
      message.props.bone_positions.buffer.slice(
        message.props.bone_positions.byteOffset,
        message.props.bone_positions.byteOffset +
          message.props.bone_positions.byteLength,
      ),
    );

    const bones: THREE.Bone[] = [];
    bonesRef.current = bones;
    for (let i = 0; i < bone_positions.length / 3; i++) {
      bones.push(new THREE.Bone());
    }
    const boneInverses: THREE.Matrix4[] = [];
    const xyzw_quat = new THREE.Quaternion();
    bones.forEach((bone, i) => {
      xyzw_quat.set(
        bone_wxyzs[i * 4 + 1],
        bone_wxyzs[i * 4 + 2],
        bone_wxyzs[i * 4 + 3],
        bone_wxyzs[i * 4 + 0],
      );

      const boneInverse = new THREE.Matrix4();
      boneInverse.makeRotationFromQuaternion(xyzw_quat);
      boneInverse.setPosition(
        bone_positions[i * 3 + 0],
        bone_positions[i * 3 + 1],
        bone_positions[i * 3 + 2],
      );
      boneInverse.invert();
      boneInverses.push(boneInverse);

      bone.setRotationFromQuaternion(xyzw_quat);
      bone.position.set(
        bone_positions[i * 3 + 0],
        bone_positions[i * 3 + 1],
        bone_positions[i * 3 + 2],
      );
    });
    const skeleton = new THREE.Skeleton(bones, boneInverses);

    geometry.setAttribute(
      "skinIndex",
      new THREE.BufferAttribute(
        new Uint16Array(
          message.props.skin_indices.buffer.slice(
            message.props.skin_indices.byteOffset,
            message.props.skin_indices.byteOffset +
              message.props.skin_indices.byteLength,
          ),
        ),
        4,
      ),
    );
    geometry.setAttribute(
      "skinWeight",
      new THREE.BufferAttribute(
        new Float32Array(
          message.props.skin_weights!.buffer.slice(
            message.props.skin_weights!.byteOffset,
            message.props.skin_weights!.byteOffset +
              message.props.skin_weights!.byteLength,
          ),
        ),
        4,
      ),
    );

    skeleton.init();
    return { geometry, skeleton };
  }, [
    message.props.vertices.buffer,
    message.props.faces.buffer,
    message.props.skin_indices.buffer,
    message.props.skin_weights?.buffer,
    message.props.bone_wxyzs.buffer,
    message.props.bone_positions.buffer,
  ]);

  // Handle initialization and cleanup
  React.useEffect(() => {
    // Return cleanup function
    return () => {
      if (skeleton) skeleton.dispose();
      if (geometry) geometry.dispose();
      if (material) material.dispose();
      const state = viewer.skinnedMeshState.current[message.name];
      state.initialized = false;
    };
  }, [skeleton, geometry, material, message.name, viewer.skinnedMeshState]);

  // Update bone transforms for animation
  useFrame(() => {
    const parentNode = viewer.nodeRefFromName.current[message.name];
    if (parentNode === undefined) return;

    const state = viewer.skinnedMeshState.current[message.name];
    const bones = bonesRef.current;
    if (skeleton !== undefined && bones !== undefined) {
      if (!state.initialized) {
        bones.forEach((bone) => {
          parentNode.add(bone);
        });
        state.initialized = true;
      }
      bones.forEach((bone, i) => {
        const wxyz = state.poses[i].wxyz;
        const position = state.poses[i].position;
        bone.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
        bone.position.set(position[0], position[1], position[2]);
      });
    }
  });

  // No need for existence checks with useMemo as they always return values

  return (
    <skinnedMesh
      ref={ref}
      geometry={geometry}
      material={material}
      skeleton={skeleton}
      castShadow={message.props.cast_shadow}
      receiveShadow={message.props.receive_shadow}
      frustumCulled={false}
    >
      <OutlinesIfHovered alwaysMounted />
    </skinnedMesh>
  );
});
