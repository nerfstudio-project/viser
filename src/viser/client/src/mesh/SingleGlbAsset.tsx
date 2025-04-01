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
  function SingleGlbAsset(message, ref: React.ForwardedRef<THREE.Group>) {
    // Load model without passing shadow settings - we'll apply them in useEffect
    const { gltf, meshes, mixerRef } = useGlbLoader(message.props.glb_data);

    // Apply shadow settings directly to the model
    React.useEffect(() => {
      if (!gltf) return;

      gltf.scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.castShadow = message.props.cast_shadow;
          obj.receiveShadow = message.props.receive_shadow;
        }
      });
    }, [gltf, message.props.cast_shadow, message.props.receive_shadow]);

    // Apply material settings to meshes
    // This is a placeholder for any future material updates needed
    // Currently, we don't need to modify materials for GLB assets since they don't have a color property
    React.useEffect(() => {
      // Dependency on meshes is kept to ensure this runs when meshes are loaded
      // In the future, material-specific properties can be updated here if needed
    }, [meshes]);

    // Update animations on each frame if mixer exists.
    useFrame((_, delta: number) => {
      mixerRef.current?.update(delta);
    });

    if (!gltf) return null;

    return (
      <group ref={ref}>
        <primitive object={gltf.scene} scale={message.props.scale} />
        {meshes.map((mesh, i) => (
          <React.Fragment key={i}>
            {createPortal(<OutlinesIfHovered alwaysMounted />, mesh)}
          </React.Fragment>
        ))}
      </group>
    );
  },
);
