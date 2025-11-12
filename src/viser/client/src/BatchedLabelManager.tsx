import React from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { Instances, Instance } from "@react-three/drei";
// @ts-ignore - troika-three-text doesn't have type definitions
import { Text as TroikaText, BatchedText } from "troika-three-text";
import { BatchedLabelManagerContext } from "./BatchedLabelManagerContext";
import { ViewerContext } from "./ViewerContext";
import {
  setupBatchedTextMaterial,
  calculateBillboardRotation,
  createRectGeometry,
  calculateBaseFontSize,
  calculateScreenSpaceScale,
  calculateAnchorOffset,
  LABEL_BACKGROUND_COLOR,
  LABEL_BACKGROUND_OPACITY,
  LABEL_BACKGROUND_PADDING_X,
  LABEL_BACKGROUND_PADDING_Y,
} from "./LabelUtils";

/**
 * Metadata stored for each text object.
 */
interface TextInfo {
  nodeName: string;
  parentName: string;
  depthTest: boolean;
  batchedText: BatchedText;
  backgroundInstanceId: string; // ID for the background rectangle instance
  backgroundDepthTest: boolean; // Depth test setting for the background
  fontSizeMode: "screen" | "scene"; // Font sizing mode
  fontScreenScale: number; // Scale factor for screen-space mode
  fontSceneHeight: number; // Font height for scene-space mode
  baseFontSize: number; // Computed base font size used for rendering
  anchorX: "left" | "center" | "right"; // Anchor X position
  anchorY: "top" | "middle" | "bottom"; // Anchor Y position
}

/**
 * Global manager for all labels using batched rendering.
 * Labels are grouped by depth_test setting for optimal performance.
 */
export const BatchedLabelManager: React.FC<{
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
  const tempWorldPos = React.useRef(new THREE.Vector3());

  // Create rectangle geometry once.
  const rectGeometry = React.useMemo(() => createRectGeometry(), []);

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
    (
      text: TroikaText,
      nodeName: string,
      depthTest: boolean,
      fontSizeMode: "screen" | "scene",
      fontScreenScale: number,
      fontSceneHeight: number,
      anchorX: "left" | "center" | "right",
      anchorY: "top" | "middle" | "bottom",
    ) => {
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

      // Create a unique ID for the background instance, encoding depthTest to avoid stale closures.
      const backgroundInstanceId = `bg-${depthTest}-${backgroundInstanceCounterRef.current++}`;

      // Create a ref for this background instance.
      backgroundInstanceRefsRef.current.set(
        backgroundInstanceId,
        React.createRef(),
      );

      // Add to background instances set (triggers re-render to create Instance component).
      setBackgroundInstances(
        (prev) => new Set([...prev, backgroundInstanceId]),
      );

      // Calculate base font size based on mode.
      const baseFontSize = calculateBaseFontSize(
        fontSizeMode,
        fontScreenScale,
        fontSceneHeight,
      );

      // Store text metadata.
      textInfoRef.current.set(text, {
        nodeName,
        parentName,
        depthTest,
        batchedText: targetBatch,
        backgroundInstanceId,
        backgroundDepthTest: depthTest,
        fontSizeMode,
        fontScreenScale,
        fontSceneHeight,
        baseFontSize,
        anchorX,
        anchorY,
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

  const updateText = React.useCallback(
    (
      text: TroikaText,
      depthTest: boolean,
      fontSizeMode: "screen" | "scene",
      fontScreenScale: number,
      fontSceneHeight: number,
      anchorX: "left" | "center" | "right",
      anchorY: "top" | "middle" | "bottom",
    ) => {
      const textInfo = textInfoRef.current.get(text);
      if (!textInfo) return;

      // Check if depthTest changed - this requires moving to a different batch.
      if (textInfo.depthTest !== depthTest) {
        // Need to do full re-registration for batch migration.
        const nodeName = textInfo.nodeName;
        unregisterText(text);
        registerText(
          text,
          nodeName,
          depthTest,
          fontSizeMode,
          fontScreenScale,
          fontSceneHeight,
          anchorX,
          anchorY,
        );
        return;
      }

      // Update metadata in place (no batch migration needed).
      const baseFontSize = calculateBaseFontSize(
        fontSizeMode,
        fontScreenScale,
        fontSceneHeight,
      );

      textInfo.fontSizeMode = fontSizeMode;
      textInfo.fontScreenScale = fontScreenScale;
      textInfo.fontSceneHeight = fontSceneHeight;
      textInfo.baseFontSize = baseFontSize;
      textInfo.anchorX = anchorX;
      textInfo.anchorY = anchorY;

      // Update text object's anchor properties and trigger sync.
      text.anchorX = anchorX;
      text.anchorY = anchorY;
      dirtyBatchesRef.current.add(textInfo.batchedText);
    },
    [registerText, unregisterText],
  );

  const contextValue = React.useMemo(
    () => ({
      registerText,
      unregisterText,
      updateText,
      syncBatchedText,
      syncText,
    }),
    [registerText, unregisterText, updateText, syncBatchedText, syncText],
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
    calculateBillboardRotation(
      group,
      camera,
      groupQuaternion.current,
      billboardQuaternion.current,
    );

    // Set material properties on BatchedText instances if not yet set.
    // BatchedText creates its material during the render loop, so we check here.
    batchedTextsRef.current.forEach((batchedText, depthTest) => {
      if (!materialPropsSetRef.current.has(batchedText)) {
        if (setupBatchedTextMaterial(batchedText, depthTest)) {
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

          // Apply font sizing based on mode.
          let paddingX = LABEL_BACKGROUND_PADDING_X;
          let paddingY = LABEL_BACKGROUND_PADDING_Y;

          if (textInfo.fontSizeMode === "screen") {
            // Scale based on distance and FOV to maintain consistent visual size.
            // Convert text position from local space to world space using group's matrix.
            tempWorldPos.current.copy(text.position);
            tempWorldPos.current.applyMatrix4(group.matrixWorld);
            const scale = calculateScreenSpaceScale(
              camera,
              tempWorldPos.current,
            );

            // Set fontSize directly - Troika applies this as a scale, no sync needed.
            // baseFontSize already includes fontScreenScale.
            text.fontSize = textInfo.baseFontSize * scale;
            // Also scale padding to maintain constant screen-space padding.
            paddingX =
              LABEL_BACKGROUND_PADDING_X * scale * textInfo.fontScreenScale;
            paddingY =
              LABEL_BACKGROUND_PADDING_Y * scale * textInfo.fontScreenScale;
          } else {
            // Use the fixed scene-space font size - no sync needed.
            text.fontSize = textInfo.baseFontSize;
          }

          // Update background instance position and scale.
          const backgroundRef = backgroundInstanceRefsRef.current.get(
            textInfo.backgroundInstanceId,
          );
          if (backgroundRef?.current && text.textRenderInfo) {
            const bg = backgroundRef.current;

            // Get text bounds from textRenderInfo.
            const bounds = text.textRenderInfo.blockBounds;
            if (bounds) {
              const [minX, minY, maxX, maxY] = bounds;
              const width = maxX - minX;
              const height = maxY - minY;

              // Calculate rectangle dimensions (text + padding).
              const rectWidth = width + paddingX;
              const rectHeight = height + paddingY;

              // Calculate rectangle bounds in text-local coordinates.
              const rectMinX = minX - paddingX / 2;
              const rectMaxX = maxX + paddingX / 2;
              const rectMinY = minY - paddingY / 2;
              const rectMaxY = maxY + paddingY / 2;

              // Calculate offset from text anchor to rectangle center.
              const { offsetX, offsetY } = calculateAnchorOffset(
                textInfo.anchorX,
                textInfo.anchorY,
                rectMinX,
                rectMaxX,
                rectMinY,
                rectMaxY,
              );

              // Position background at text center.
              // The center offset is in local space, so we need to rotate it by the billboard quaternion.
              localOffset.current.set(offsetX, offsetY, 0);
              localOffset.current.applyQuaternion(billboardQuaternion.current);

              bg.position.copy(text.position);
              bg.position.add(localOffset.current);

              // Scale background to rectangle size.
              bg.scale.set(rectWidth, rectHeight, 1);

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
          const backgroundRef = backgroundInstanceRefsRef.current.get(
            textInfo.backgroundInstanceId,
          );
          if (backgroundRef?.current) {
            backgroundRef.current.visible = false;
          }
        }

        // Apply billboard rotation.
        text.quaternion.copy(billboardQuaternion.current);
      } catch (error) {
        console.error("[BatchedLabelManager] Error updating text:", error);
      }
    });
  });

  // Split backgrounds by depth test setting.
  // Parse depthTest from ID to avoid stale closure issues.
  const backgroundsByDepthTest = React.useMemo(() => {
    const map = new Map<boolean, string[]>();
    map.set(true, []);
    map.set(false, []);

    backgroundInstances.forEach((instanceId) => {
      // ID format: "bg-{depthTest}-{counter}"
      const depthTest = instanceId.startsWith("bg-true");
      const list = map.get(depthTest);
      if (list) {
        list.push(instanceId);
      }
    });

    return map;
  }, [backgroundInstances]);

  return (
    <BatchedLabelManagerContext.Provider value={contextValue}>
      {group && <primitive object={group} />}
      {/* Background rectangles with depth test = true */}
      <Instances frustumCulled={false}>
        <primitive object={rectGeometry} attach="geometry" />
        <meshBasicMaterial
          color={LABEL_BACKGROUND_COLOR}
          transparent={true}
          opacity={LABEL_BACKGROUND_OPACITY}
          depthTest={true}
          depthWrite={false}
          toneMapped={false}
        />
        {backgroundsByDepthTest.get(true)?.map((instanceId) => {
          const ref = backgroundInstanceRefsRef.current.get(instanceId);
          return <Instance key={instanceId} ref={ref} renderOrder={9999} />;
        })}
      </Instances>
      {/* Background rectangles with depth test = false */}
      <Instances frustumCulled={false}>
        <primitive object={rectGeometry} attach="geometry" />
        <meshBasicMaterial
          color={LABEL_BACKGROUND_COLOR}
          transparent={true}
          opacity={LABEL_BACKGROUND_OPACITY}
          depthTest={false}
          depthWrite={false}
          toneMapped={false}
        />
        {backgroundsByDepthTest.get(false)?.map((instanceId) => {
          const ref = backgroundInstanceRefsRef.current.get(instanceId);
          return <Instance key={instanceId} ref={ref} renderOrder={9999} />;
        })}
      </Instances>
      {children}
    </BatchedLabelManagerContext.Provider>
  );
};
