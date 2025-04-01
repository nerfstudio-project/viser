import React from "react";
import * as THREE from "three";
import { createStandardMaterial, rgbToInt } from "./MeshUtils";
import { MeshMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";

/**
 * Component for rendering basic THREE.js meshes
 */
export const BasicMesh = React.forwardRef<THREE.Mesh, MeshMessage>(
  function BasicMesh(message, ref: React.ForwardedRef<THREE.Mesh>) {
    // Setup a basic material just once - we'll update all properties via direct methods
    const material = React.useMemo(
      () => {
        // Create a basic material with default properties - all will be updated in useEffect
        return createStandardMaterial({
          material: "standard", // Will be updated if different
          color: [128, 128, 128], // Will be updated immediately
          wireframe: false, // Will be updated
          opacity: null, // Will be updated
          flat_shading: false, // Will be updated
          side: "front", // Will be updated
        });
      },
      [
        // No dependencies - we never want to recreate the material
        // All properties will be updated via direct mutation in useEffect
      ],
    );

    // Setup geometry using memoization.
    const geometry = React.useMemo(() => {
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

    // Update material properties directly without recreating the material
    React.useEffect(() => {
      // Generate gradient map for toon materials if needed
      const generateGradientMap = (shades: 3 | 5): THREE.DataTexture => {
        const texture = new THREE.DataTexture(
          Uint8Array.from(
            shades === 3 ? [0, 128, 255] : [0, 64, 128, 192, 255],
          ),
          shades,
          1,
          THREE.RedFormat,
        );
        texture.needsUpdate = true;
        return texture;
      };

      // Convert material type if needed
      if (
        (message.props.material === "toon3" ||
          message.props.material === "toon5") &&
        !(material instanceof THREE.MeshToonMaterial)
      ) {
        // Need to replace material entirely for type changes
        const newMaterial = new THREE.MeshToonMaterial({
          gradientMap: generateGradientMap(
            message.props.material === "toon3" ? 3 : 5,
          ),
        });

        // Copy properties from old material
        newMaterial.color.setHex(rgbToInt(message.props.color));

        // Replace the material
        if (ref && typeof ref !== "function" && ref.current) {
          ref.current.material = newMaterial;
          material.dispose();
        }
      } else if (
        message.props.material === "standard" &&
        !(material instanceof THREE.MeshStandardMaterial)
      ) {
        // Need to replace material entirely for type changes
        const newMaterial = new THREE.MeshStandardMaterial();

        // Copy properties from old material
        newMaterial.color.setHex(rgbToInt(message.props.color));

        // Replace the material
        if (ref && typeof ref !== "function" && ref.current) {
          ref.current.material = newMaterial;
          material.dispose();
        }
      } else {
        // Update existing material properties
        if ("color" in material && material.color instanceof THREE.Color) {
          material.color.setHex(rgbToInt(message.props.color));
        }

        if ("wireframe" in material) {
          material.wireframe = message.props.wireframe;
        }

        if ("opacity" in material) {
          material.opacity = message.props.opacity ?? 1.0;
          material.transparent = message.props.opacity !== null;
        }

        if ("flatShading" in material) {
          material.flatShading =
            message.props.flat_shading && !message.props.wireframe;
          material.needsUpdate = true;
        }

        if ("side" in material) {
          material.side = {
            front: THREE.FrontSide,
            back: THREE.BackSide,
            double: THREE.DoubleSide,
          }[message.props.side];
        }

        // Update gradient map for toon materials
        if (material instanceof THREE.MeshToonMaterial) {
          if (message.props.material === "toon3") {
            material.gradientMap = generateGradientMap(3);
          } else if (message.props.material === "toon5") {
            material.gradientMap = generateGradientMap(5);
          }
          material.needsUpdate = true;
        }
      }
    }, [
      material,
      ref,
      message.props.material,
      message.props.color,
      message.props.wireframe,
      message.props.opacity,
      message.props.flat_shading,
      message.props.side,
    ]);

    // Clean up resources when component unmounts.
    React.useEffect(() => {
      return () => {
        if (material) material.dispose();
        if (geometry) geometry.dispose();
      };
    }, [material, geometry]);

    return (
      <mesh
        ref={ref}
        geometry={geometry}
        material={material}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
      >
        <OutlinesIfHovered alwaysMounted />
      </mesh>
    );
  },
);
