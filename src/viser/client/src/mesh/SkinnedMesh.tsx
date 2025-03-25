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

  // Create persistent geometry and material
  const [material, setMaterial] = React.useState<THREE.Material>();
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [skeleton, setSkeleton] = React.useState<THREE.Skeleton>();
  const bonesRef = React.useRef<THREE.Bone[]>();

  // Setup material
  React.useEffect(() => {
    const material = createStandardMaterial(message.props);
    setMaterial(material);

    return () => {
      // Dispose material when done
      material.dispose();
    };
  }, [
    message.props.material,
    message.props.color,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
  ]);

  // Setup geometry and skeleton
  React.useEffect(() => {
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
    setGeometry(geometry);
    setSkeleton(skeleton);

    return () => {
      skeleton.dispose();
      geometry.dispose();
      const state = viewer.skinnedMeshState.current[message.name];
      state.initialized = false;
    };
  }, [
    message.name,
    message.props.vertices.buffer,
    message.props.faces.buffer,
    message.props.skin_indices.buffer,
    message.props.skin_weights?.buffer,
    message.props.bone_wxyzs.buffer,
    message.props.bone_positions.buffer,
    viewer.skinnedMeshState,
  ]);

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

  if (
    geometry === undefined ||
    material === undefined ||
    skeleton === undefined
  ) {
    return null;
  }

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
