import React from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { Instances, Instance } from "@react-three/drei";
// @ts-ignore - troika-three-text doesn't have type definitions
import { Text as TroikaText, BatchedText } from "troika-three-text";
import { BatchedTextManagerContext } from "./BatchedTextManagerContext";
import { ViewerContext } from "./ViewerContext";

/**
 * Metadata stored for each text object.
 */
interface TextInfo {
  nodeName: string;
  parentName: string;
  depthTest: boolean;
  batchedText: BatchedText;
  backgroundInstanceId: string; // ID for the background rectangle instance
}

/**
 * Global manager for all text labels using batched rendering.
 * Labels are grouped by depth_test setting for optimal performance.
 */
export const GlobalBatchedTextManager: React.FC<{
  children?: React.ReactNode;
}> = ({ children }) => {
  const viewer = React.useContext(ViewerContext)!;
  const [group, setGroup] = React.useState<THREE.Group | null>(null);

  // One BatchedText instance per depthTest setting (true/false).
  const batchedTextsRef = React.useRef<Map<boolean, BatchedText>>(new Map());
  const textInfoRef = React.useRef(new Map<TroikaText, TextInfo>());
  const materialPropsSetRef = React.useRef(new Set<BatchedText>());

  // Background rectangle instances for each text label.
  const [backgroundInstances, setBackgroundInstances] = React.useState<
    Set<string>
  >(new Set());
  const backgroundInstanceRefsRef = React.useRef<
    Map<string, React.RefObject<THREE.Object3D>>
  >(new Map());

  // Dirty flags for batching sync() calls - synced in useFrame.
  const dirtyBatchesRef = React.useRef(new Set<BatchedText>());

  // Counter for generating unique background instance IDs.
  const backgroundInstanceCounterRef = React.useRef(0);

  // Reuse objects to avoid allocations.
  const groupQuaternion = React.useRef(new THREE.Quaternion());
  const billboardQuaternion = React.useRef(new THREE.Quaternion());
  const frustum = React.useRef(new THREE.Frustum());
  const projScreenMatrix = React.useRef(new THREE.Matrix4());
  const localOffset = React.useRef(new THREE.Vector3());

  // Create rounded rectangle geometry once.
  const roundedRectGeometry = React.useMemo(() => {
    const width = 1;
    const height = 1;
    const radius = 0.1; // Corner radius as fraction of unit square
    const shape = new THREE.Shape();

    // Start at top-left corner
    shape.moveTo(-width / 2 + radius, height / 2);
    // Top edge
    shape.lineTo(width / 2 - radius, height / 2);
    // Top-right corner
    shape.quadraticCurveTo(width / 2, height / 2, width / 2, height / 2 - radius);
    // Right edge
    shape.lineTo(width / 2, -height / 2 + radius);
    // Bottom-right corner
    shape.quadraticCurveTo(width / 2, -height / 2, width / 2 - radius, -height / 2);
    // Bottom edge
    shape.lineTo(-width / 2 + radius, -height / 2);
    // Bottom-left corner
    shape.quadraticCurveTo(-width / 2, -height / 2, -width / 2, -height / 2 + radius);
    // Left edge
    shape.lineTo(-width / 2, height / 2 - radius);
    // Top-left corner
    shape.quadraticCurveTo(-width / 2, height / 2, -width / 2 + radius, height / 2);

    return new THREE.ShapeGeometry(shape);
  }, []);

  // Create the group once on mount.
  React.useEffect(() => {
    const newGroup = new THREE.Group();
    setGroup(newGroup);

    // Cleanup on unmount.
    return () => {
      batchedTextsRef.current.forEach((batchedText) => {
        newGroup.remove(batchedText);
        batchedText.dispose();
      });
    };
  }, []);

  // API for components to register/unregister their text objects.
  const registerText = React.useCallback(
    (text: TroikaText, nodeName: string, depthTest: boolean) => {
      if (!group) return;

      // Get or create BatchedText for this depthTest setting.
      let targetBatch = batchedTextsRef.current.get(depthTest);
      if (!targetBatch) {
        targetBatch = new BatchedText();
        targetBatch.renderOrder = 10_000;
        batchedTextsRef.current.set(depthTest, targetBatch);
        group.add(targetBatch);
      }

      targetBatch.add(text);

      // Mark batch as needing sync - will be synced in useFrame.
      dirtyBatchesRef.current.add(targetBatch);

      // Compute parent name once and store it.
      const parentName = nodeName.split("/").slice(0, -1).join("/");

      // Create a unique ID for the background instance.
      const backgroundInstanceId = `bg-${backgroundInstanceCounterRef.current++}`;

      // Create a ref for this background instance.
      backgroundInstanceRefsRef.current.set(
        backgroundInstanceId,
        React.createRef(),
      );

      // Add to background instances set (triggers re-render to create Instance component).
      setBackgroundInstances((prev) => new Set([...prev, backgroundInstanceId]));

      // Store text metadata.
      textInfoRef.current.set(text, {
        nodeName,
        parentName,
        depthTest,
        batchedText: targetBatch,
        backgroundInstanceId,
      });
    },
    [group],
  );

  const unregisterText = React.useCallback((text: TroikaText) => {
    const textInfo = textInfoRef.current.get(text);
    if (textInfo) {
      textInfo.batchedText.remove(text);

      // Mark batch as needing sync - will be synced in useFrame.
      dirtyBatchesRef.current.add(textInfo.batchedText);

      // Remove background instance.
      backgroundInstanceRefsRef.current.delete(textInfo.backgroundInstanceId);
      setBackgroundInstances((prev) => {
        const next = new Set(prev);
        next.delete(textInfo.backgroundInstanceId);
        return next;
      });

      textInfoRef.current.delete(text);
    }
  }, []);

  const syncBatchedText = React.useCallback(() => {
    // Mark all batches as needing sync - will be synced in useFrame.
    batchedTextsRef.current.forEach((batch) => {
      dirtyBatchesRef.current.add(batch);
    });
  }, []);

  const syncText = React.useCallback((text: TroikaText) => {
    // Mark only the batch containing this specific text as needing sync.
    const textInfo = textInfoRef.current.get(text);
    if (textInfo) {
      dirtyBatchesRef.current.add(textInfo.batchedText);
    }
  }, []);

  const contextValue = React.useMemo(
    () => ({ registerText, unregisterText, syncBatchedText, syncText }),
    [registerText, unregisterText, syncBatchedText, syncText],
  );

  // Billboard rotation, position updates, and visibility culling in render loop.
  useFrame(({ camera }) => {
    if (!group) return;

    // Sync dirty batches (batches multiple add/remove/update calls).
    dirtyBatchesRef.current.forEach((batchedText) => {
      batchedText.sync();
    });
    dirtyBatchesRef.current.clear();

    // Compute frustum for culling (shared across all batches).
    projScreenMatrix.current.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    );
    frustum.current.setFromProjectionMatrix(projScreenMatrix.current);

    // Calculate billboard rotation accounting for parent transform (shared).
    group.updateMatrix();
    group.updateWorldMatrix(false, false);
    group.getWorldQuaternion(groupQuaternion.current);
    camera
      .getWorldQuaternion(billboardQuaternion.current)
      .premultiply(groupQuaternion.current.invert());

    // Set material properties on BatchedText instances if not yet set.
    // BatchedText creates its material during the render loop, so we check here.
    batchedTextsRef.current.forEach((batchedText, depthTest) => {
      if (!materialPropsSetRef.current.has(batchedText)) {
        const material = batchedText.material;
        if (material) {
          // Material can be an array [outlineMaterial, mainMaterial] or a single material.
          const materials = Array.isArray(material) ? material : [material];
          materials.forEach((mat) => {
            // Set depth test based on batch setting.
            mat.depthTest = depthTest;
            // Always disable depthWrite to avoid z-fighting between outline and fill.
            mat.depthWrite = false;
            // Mark as transparent for proper alpha blending and depth sorting.
            mat.transparent = true;
            mat.needsUpdate = true;
          });
          materialPropsSetRef.current.add(batchedText);
        }
      }
    });

    // Update text positions, visibility, and apply billboard rotation.
    // Iterate over our own textInfoRef map instead of relying on private _members API.
    textInfoRef.current.forEach((textInfo, text) => {
      try {
        // Compute position from parent node + local position.
        const parentObj =
          viewer.mutable.current.nodeRefFromName[textInfo.parentName];
        const node = viewer.useSceneTree.getState()[textInfo.nodeName];

        if (parentObj && node) {
          // Get parent's world position.
          parentObj.getWorldPosition(text.position);

          // Add label's local position offset.
          const localPos = node.position ?? [0, 0, 0];
          text.position.x += localPos[0];
          text.position.y += localPos[1];
          text.position.z += localPos[2];

          // Use effectiveVisibility which includes parent chain.
          // Use opacity instead of visible since BatchedText ignores individual text.visible.
          const isVisible = node.effectiveVisibility ?? false;
          if (isVisible) {
            text.fillOpacity = 1.0;
          } else {
            text.fillOpacity = 0.0;
          }

          // Update background instance position and scale.
          const backgroundRef =
            backgroundInstanceRefsRef.current.get(textInfo.backgroundInstanceId);
          if (backgroundRef?.current && text.textRenderInfo) {
            const bg = backgroundRef.current;

            // Ensure text measurements are up to date.
            text.sync();

            // Get text bounds from textRenderInfo.
            const bounds = text.textRenderInfo.blockBounds;
            if (bounds) {
              const [minX, minY, maxX, maxY] = bounds;
              const width = maxX - minX;
              const height = maxY - minY;
              const centerX = (minX + maxX) / 2;
              const centerY = (minY + maxY) / 2;

              // Position background at text center.
              // The center offset is in local space, so we need to rotate it by the billboard quaternion.
              localOffset.current.set(centerX, centerY, 0);
              localOffset.current.applyQuaternion(billboardQuaternion.current);

              bg.position.copy(text.position);
              bg.position.add(localOffset.current);

              // Scale background to text size with padding.
              const paddingX = 0.02;
              const paddingY = 0.01;
              bg.scale.set(width + paddingX, height + paddingY, 1);

              // Match text rotation.
              bg.quaternion.copy(billboardQuaternion.current);

              // Match visibility.
              bg.visible = isVisible;
            } else {
              bg.visible = false;
            }
          }
        } else {
          text.fillOpacity = 0.0;

          // Hide background when text is hidden.
          const backgroundRef =
            backgroundInstanceRefsRef.current.get(textInfo.backgroundInstanceId);
          if (backgroundRef?.current) {
            backgroundRef.current.visible = false;
          }
        }

        // Apply billboard rotation.
        text.quaternion.copy(billboardQuaternion.current);
      } catch (error) {
        console.error("[GlobalBatchedTextManager] Error updating text:", error);
      }
    });
  });

  return (
    <BatchedTextManagerContext.Provider value={contextValue}>
      {group && <primitive object={group} />}
      {/* Background rectangles for text labels */}
      <Instances frustumCulled={false}>
        <primitive object={roundedRectGeometry} attach="geometry" />
        <meshBasicMaterial
          color={0xffffff}
          transparent={true}
          opacity={0.9}
          depthTest={true}
          depthWrite={false}
          toneMapped={false}
        />
        {Array.from(backgroundInstances).map((instanceId) => {
          const ref = backgroundInstanceRefsRef.current.get(instanceId);
          return <Instance key={instanceId} ref={ref} renderOrder={9999} />;
        })}
      </Instances>
      {children}
    </BatchedTextManagerContext.Provider>
  );
};
