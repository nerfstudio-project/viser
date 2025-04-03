import * as THREE from "three";
import {
  MeshBasicMaterial,
  MeshDepthMaterial,
  MeshDistanceMaterial,
  MeshLambertMaterial,
  MeshMatcapMaterial,
  MeshNormalMaterial,
  MeshPhongMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  MeshToonMaterial,
  ShadowMaterial,
  SpriteMaterial,
  RawShaderMaterial,
  ShaderMaterial,
  PointsMaterial,
  LineBasicMaterial,
  LineDashedMaterial,
} from "three";

/**
 * Type definition for all possible Three.js materials
 */
export type AllPossibleThreeJSMaterials =
  | MeshBasicMaterial
  | MeshDepthMaterial
  | MeshDistanceMaterial
  | MeshLambertMaterial
  | MeshMatcapMaterial
  | MeshNormalMaterial
  | MeshPhongMaterial
  | MeshPhysicalMaterial
  | MeshStandardMaterial
  | MeshToonMaterial
  | ShadowMaterial
  | SpriteMaterial
  | RawShaderMaterial
  | ShaderMaterial
  | PointsMaterial
  | LineBasicMaterial
  | LineDashedMaterial;

/**
 * Convert RGB array to integer color representation
 */
export function rgbToInt(rgb: [number, number, number]): number {
  return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
}

/**
 * Create a gradient texture for toon materials
 */
export function generateGradientMap(shades: 3 | 5): THREE.DataTexture {
  const texture = new THREE.DataTexture(
    Uint8Array.from(shades === 3 ? [0, 128, 255] : [0, 64, 128, 192, 255]),
    shades,
    1,
    THREE.RedFormat,
  );

  texture.needsUpdate = true;
  return texture;
}

/**
 * Helper function for type checking
 */
export function assertUnreachable(x: never): never {
  throw new Error(`Should never get here! ${x}`);
}

/**
 * Helper to dispose material textures
 */
export function disposeMaterial(material: AllPossibleThreeJSMaterials) {
  if ("map" in material) material.map?.dispose();
  if ("lightMap" in material) material.lightMap?.dispose();
  if ("bumpMap" in material) material.bumpMap?.dispose();
  if ("normalMap" in material) material.normalMap?.dispose();
  if ("specularMap" in material) material.specularMap?.dispose();
  if ("envMap" in material) material.envMap?.dispose();
  if ("alphaMap" in material) material.alphaMap?.dispose();
  if ("aoMap" in material) material.aoMap?.dispose();
  if ("displacementMap" in material) material.displacementMap?.dispose();
  if ("emissiveMap" in material) material.emissiveMap?.dispose();
  if ("gradientMap" in material) material.gradientMap?.dispose();
  if ("metalnessMap" in material) material.metalnessMap?.dispose();
  if ("roughnessMap" in material) material.roughnessMap?.dispose();
  material.dispose(); // disposes any programs associated with the material
}

/**
 * Create standard material based on properties
 */
export function createStandardMaterial(props: {
  material: "standard" | "toon3" | "toon5";
  color: [number, number, number];
  wireframe: boolean;
  opacity: number | null;
  flat_shading: boolean;
  side: "front" | "back" | "double";
}): THREE.Material {
  const standardArgs = {
    color: rgbToInt(props.color),
    wireframe: props.wireframe,
    transparent: props.opacity !== null,
    opacity: props.opacity ?? 1.0,
    // Flat shading only makes sense for non-wireframe materials.
    flatShading: props.flat_shading && !props.wireframe,
    side: {
      front: THREE.FrontSide,
      back: THREE.BackSide,
      double: THREE.DoubleSide,
    }[props.side],
  };

  if (props.material == "standard" || props.wireframe) {
    return new THREE.MeshStandardMaterial(standardArgs);
  } else if (props.material == "toon3") {
    return new THREE.MeshToonMaterial({
      gradientMap: generateGradientMap(3),
      ...standardArgs,
    });
  } else if (props.material == "toon5") {
    return new THREE.MeshToonMaterial({
      gradientMap: generateGradientMap(5),
      ...standardArgs,
    });
  } else {
    return assertUnreachable(props.material);
  }
}
