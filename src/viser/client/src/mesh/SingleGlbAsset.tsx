import React from "react";
import * as THREE from "three";
import { createPortal } from "@react-three/fiber";
import { GlbMessage } from "../WebsocketMessages";
import { useGlbLoader } from "./GlbLoaderUtils";
import { OutlinesIfHovered } from "../OutlinesIfHovered";
import { useFrame } from "@react-three/fiber";

/**
 * Component for rendering a single GLB model
 */
export const SingleGlbAsset = React.forwardRef<THREE.Group, GlbMessage>(
  function SingleGlbAsset(message, ref) {
    const { gltf, meshes, mixerRef } = useGlbLoader(
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

    if (!gltf) return null;

    return (
      <group ref={ref}>
        <primitive
          object={gltf.scene}
          scale={message.props.scale}
          castShadow={message.props.cast_shadow}
          receiveShadow={message.props.receive_shadow}
        />
        {meshes.map((mesh, i) => (
          <React.Fragment key={i}>
            {createPortal(<OutlinesIfHovered alwaysMounted />, mesh)}
          </React.Fragment>
        ))}
      </group>
    );
  },
);
