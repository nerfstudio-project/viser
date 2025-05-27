import React, { useMemo } from "react";
import * as THREE from "three";
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
  BatchedMeshesMessage
>(function BatchedMesh(message, ref) {
  const viewer = React.useContext(ViewerContext)!;
  const clickable =
    viewer.useSceneTree(
      (state) => state.nodeFromName[message.name]?.clickable,
    ) ?? false;

  // Create a material based on the message props.
  const material = useMemo(() => {
    // Create the material with properties from the message.
    const mat = createStandardMaterial({
      material: message.props.material,
      color: message.props.color,
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
    message.props.color,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
  ]);

  // Setup geometry using memoization.
  const geometry = useMemo(() => {
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
    return geometry;
  }, [message.props.vertices.buffer, message.props.faces.buffer]);

  return (
    <BatchedMeshBase
      ref={ref}
      geometry={geometry}
      material={material}
      batched_positions={message.props.batched_positions}
      batched_wxyzs={message.props.batched_wxyzs}
      batched_scales={message.props.batched_scales}
      lod={message.props.lod}
      cast_shadow={message.props.cast_shadow}
      receive_shadow={message.props.receive_shadow}
      clickable={clickable}
    />
  );
});
