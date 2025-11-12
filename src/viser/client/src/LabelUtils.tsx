import * as THREE from "three";
// @ts-ignore - troika-three-text doesn't have type definitions
import { BatchedText } from "troika-three-text";

/**
 * Shared configuration for label text styling.
 */
export const LABEL_FONT = "./Inter-VariableFont_slnt,wght.ttf";
export const LABEL_TEXT_COLOR = 0x000000; // Black
export const LABEL_SDF_GLYPH_SIZE = 32;

/**
 * Shared configuration for label backgrounds.
 */
export const LABEL_BACKGROUND_COLOR = 0xffffff; // White
export const LABEL_BACKGROUND_OPACITY = 0.9;
export const LABEL_BACKGROUND_PADDING_X = 0.08;
export const LABEL_BACKGROUND_PADDING_Y = 0.005;

/**
 * Set material properties on a BatchedText instance.
 * This must be called in useFrame after the material is created.
 */
export function setupBatchedTextMaterial(
  batchedText: BatchedText,
  depthTest: boolean,
): boolean {
  const material = batchedText.material;
  if (!material) return false;

  // Material can be an array [outlineMaterial, mainMaterial] or a single material.
  const materials = Array.isArray(material) ? material : [material];
  materials.forEach((mat) => {
    // Set depth test based on setting.
    mat.depthTest = depthTest;
    // Always disable depthWrite to avoid z-fighting between outline and fill.
    mat.depthWrite = false;
    // Mark as transparent for proper alpha blending and depth sorting.
    mat.transparent = true;
    mat.needsUpdate = true;
  });

  batchedText.renderOrder = 10_000;
  return true;
}

/**
 * Calculate billboard rotation for a group, accounting for parent transforms.
 * Returns the quaternion that should be applied to text objects for billboarding.
 */
export function calculateBillboardRotation(
  group: THREE.Group,
  camera: THREE.Camera,
  groupQuaternionOut: THREE.Quaternion,
  billboardQuaternionOut: THREE.Quaternion,
): THREE.Quaternion {
  group.updateMatrix();
  group.updateWorldMatrix(false, false);
  group.getWorldQuaternion(groupQuaternionOut);

  camera
    .getWorldQuaternion(billboardQuaternionOut)
    .premultiply(groupQuaternionOut.invert());

  return billboardQuaternionOut;
}

/**
 * Create a rectangle geometry for label backgrounds.
 */
export function createRectGeometry(
  width: number = 1,
  height: number = 1,
): THREE.PlaneGeometry {
  return new THREE.PlaneGeometry(width, height);
}

/**
 * Calculate base font size from sizing mode and parameters.
 */
export function calculateBaseFontSize(
  mode: "screen" | "scene",
  screenScale: number,
  sceneHeight: number,
): number {
  return mode === "screen" ? 0.3 * screenScale : sceneHeight;
}

/**
 * Calculate screen-space scale factor for labels.
 * Returns scale factor to apply to base font size.
 */
export function calculateScreenSpaceScale(
  camera: THREE.Camera,
  worldPosition: THREE.Vector3,
): number {
  if ("fov" in camera && typeof camera.fov === "number") {
    // PerspectiveCamera: use Euclidean distance and FOV.
    const distance = camera.position.distanceTo(worldPosition);
    const fovScale = Math.tan(
      ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 360,
    );
    // Reference distance is 10 units (baseFontSize is calibrated for this).
    return (distance / 10.0) * fovScale;
  } else {
    // OrthographicCamera: use constant scale (no perspective).
    return 1.0;
  }
}

/**
 * Calculate anchor offset for a rectangle.
 * Returns the offset from the text anchor point (0,0) to the rectangle center.
 */
export function calculateAnchorOffset(
  anchorX: "left" | "center" | "right",
  anchorY: "top" | "middle" | "bottom",
  rectMinX: number,
  rectMaxX: number,
  rectMinY: number,
  rectMaxY: number,
): { offsetX: number; offsetY: number } {
  // Calculate anchor point on rectangle based on user's anchor choice.
  let rectAnchorX = 0;
  let rectAnchorY = 0;

  if (anchorX === "left") {
    rectAnchorX = rectMinX;
  } else if (anchorX === "right") {
    rectAnchorX = rectMaxX;
  } else {
    // center
    rectAnchorX = (rectMinX + rectMaxX) / 2;
  }

  if (anchorY === "top") {
    rectAnchorY = rectMaxY;
  } else if (anchorY === "bottom") {
    rectAnchorY = rectMinY;
  } else {
    // middle
    rectAnchorY = (rectMinY + rectMaxY) / 2;
  }

  // Background is positioned at its center.
  const rectCenterX = (rectMinX + rectMaxX) / 2;
  const rectCenterY = (rectMinY + rectMaxY) / 2;

  // Offset from text anchor (0, 0) to rectangle center.
  return {
    offsetX: rectCenterX - rectAnchorX,
    offsetY: rectCenterY - rectAnchorY,
  };
}
