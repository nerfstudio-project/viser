/**
 * Custom environment component using HDR JPEG (gainmap) format.
 * This replaces drei's Environment component to use the smaller HDR JPEG files.
 */

import { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { HDRJPGLoader } from "@monogrid/gainmap-js";

interface HDRJPGEnvironmentProps {
  /** Path to the HDR JPEG file. */
  files: string;
  /** Whether to use the HDRI as scene background. */
  background?: boolean;
  /** Blurriness of the background (0 = sharp, 1 = fully blurred). */
  backgroundBlurriness?: number;
  /** Intensity of the background. */
  backgroundIntensity?: number;
  /** Rotation of the background as Euler angles. */
  backgroundRotation?: THREE.Euler;
  /** Intensity of the environment lighting. */
  environmentIntensity?: number;
  /** Rotation of the environment lighting as Euler angles. */
  environmentRotation?: THREE.Euler;
}

export function HDRJPGEnvironment({
  files,
  background = false,
  backgroundBlurriness = 0,
  backgroundIntensity = 1,
  backgroundRotation,
  environmentIntensity = 1,
  environmentRotation,
}: HDRJPGEnvironmentProps) {
  const gl = useThree((state) => state.gl);
  const scene = useThree((state) => state.scene);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  // Load the HDR JPEG file.
  useEffect(() => {
    const loader = new HDRJPGLoader(gl);
    let disposed = false;

    loader.load(
      files,
      (result) => {
        if (disposed) {
          result.renderTarget.dispose();
          return;
        }
        const tex = result.renderTarget.texture;
        tex.mapping = THREE.EquirectangularReflectionMapping;
        setTexture(tex);
      },
      undefined,
      (error) => {
        console.error("Failed to load HDR JPEG:", error);
      },
    );

    return () => {
      disposed = true;
    };
  }, [files, gl]);

  // Apply environment map to scene.
  useEffect(() => {
    if (!texture) return;

    const prevEnvironment = scene.environment;
    const prevBackground = scene.background;
    const prevBackgroundBlurriness = scene.backgroundBlurriness;
    const prevBackgroundIntensity = scene.backgroundIntensity;
    const prevBackgroundRotation = scene.backgroundRotation?.clone();
    const prevEnvironmentIntensity = scene.environmentIntensity;
    const prevEnvironmentRotation = scene.environmentRotation?.clone();

    // Set environment.
    scene.environment = texture;
    scene.environmentIntensity = environmentIntensity;
    if (environmentRotation) {
      scene.environmentRotation = environmentRotation;
    }

    // Set background if enabled.
    if (background) {
      scene.background = texture;
      scene.backgroundBlurriness = backgroundBlurriness;
      scene.backgroundIntensity = backgroundIntensity;
      if (backgroundRotation) {
        scene.backgroundRotation = backgroundRotation;
      }
    }

    return () => {
      scene.environment = prevEnvironment;
      scene.environmentIntensity = prevEnvironmentIntensity;
      if (prevEnvironmentRotation) {
        scene.environmentRotation = prevEnvironmentRotation;
      }
      if (background) {
        scene.background = prevBackground;
        scene.backgroundBlurriness = prevBackgroundBlurriness;
        scene.backgroundIntensity = prevBackgroundIntensity;
        if (prevBackgroundRotation) {
          scene.backgroundRotation = prevBackgroundRotation;
        }
      }
    };
  }, [
    texture,
    scene,
    background,
    backgroundBlurriness,
    backgroundIntensity,
    backgroundRotation,
    environmentIntensity,
    environmentRotation,
  ]);

  return null;
}
