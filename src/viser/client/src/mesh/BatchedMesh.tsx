/** @jsxImportSource react */
import React, { useMemo } from "react";
import * as THREE from "three";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { createStandardMaterial } from "./MeshUtils";
import { BatchedMeshesMessage } from "../WebsocketMessages";
import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { ViewerContext } from "../ViewerContext";
import { BatchedMeshBase } from "./BatchedMeshBase";

/**
 * Component for rendering batched/instanced meshes
 */
export const BatchedMesh = React.forwardRef<
  InstancedMesh2,
  BatchedMeshesMessage & { children?: React.ReactNode }
>(function BatchedMesh({ children, ...message }, ref) {
  const viewer = React.useContext(ViewerContext)!;
  const clickable =
    viewer.useSceneTree((state) => state[message.name]?.clickable) ?? false;

  // Create a material based on the message props.
  const material = useMemo(() => {
    // Create the material with properties from the message.
    const mat = createStandardMaterial({
      material: message.props.material,
      wireframe: message.props.wireframe,
      opacity: message.props.opacity,
      flat_shading: message.props.flat_shading,
      side: message.props.side,
    });

    // Set additional properties.
    if (message.props.opacity !== null && message.props.opacity < 1.0) {
      mat.transparent = true;
    }

    return mat;
  }, [
    message.props.material,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
  ]);

  // Setup geometry using memoization.
  const geometry = useMemo(() => {
    let geometry = new THREE.BufferGeometry();
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

    geometry = BufferGeometryUtils.mergeVertices(geometry);

    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }, [message.props.vertices.buffer, message.props.faces.buffer]);

  return (
    // @ts-ignore - react-three-fiber JSX elements not recognized by TypeScript
    <group ref={ref}>
      <BatchedMeshBase
        geometry={geometry}
        material={material}
        batched_positions={message.props.batched_positions}
        batched_wxyzs={message.props.batched_wxyzs}
        batched_scales={message.props.batched_scales}
        batched_colors={message.props.batched_colors}
        lod={message.props.lod}
        cast_shadow={message.props.cast_shadow}
        receive_shadow={message.props.receive_shadow}
        clickable={clickable}
      />
      {children}
      {/* @ts-ignore */}
    </group>
  );
});
