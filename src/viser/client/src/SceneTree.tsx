import { useCursor } from "@react-three/drei";
import { createPortal, useFrame } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { ViewerContext } from "./App";
import { makeThrottledMessageSender } from "./WebsocketFunctions";
import { Html } from "@react-three/drei";
import { Select } from "@react-three/postprocessing";
import { immerable } from "immer";
import { Text } from "@mantine/core";
import { useSceneTreeState } from "./SceneTreeState";

export type MakeObject<T extends THREE.Object3D = THREE.Object3D> = (
  ref: React.Ref<T>,
) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode<T extends THREE.Object3D = THREE.Object3D> {
  [immerable] = true;

  public children: string[];
  public clickable: boolean;

  constructor(
    public name: string,
    public makeObject: MakeObject<T>,
    public cleanup?: () => void,
    /** unmountWhenInvisible is used to unmount <Html /> components when they
     * should be hidden.
     *
     * https://github.com/pmndrs/drei/issues/1323
     */
    public unmountWhenInvisible?: true,
  ) {
    this.children = [];
    this.clickable = false;
  }
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

function SceneNodeThreeChildren(props: {
  name: string;
  parent: THREE.Object3D;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const children = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.children,
  );

  // Create a group of children inside of the parent object.
  return createPortal(
    <group>
      {children &&
        children.map((child_id) => {
          return (
            <SceneNodeThreeObject
              key={child_id}
              name={child_id}
              parent={props.parent}
            />
          );
        })}
      <SceneNodeLabel name={props.name} />
    </group>,
    props.parent,
  );
}

/** Component for updating attributes of a scene node. */
function SceneNodeLabel(props: { name: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const labelVisible = viewer.useSceneTree(
    (state) => state.labelVisibleFromName[props.name],
  );
  return labelVisible ? (
    <Html>
      <Text
        style={{
          backgroundColor: "rgba(240, 240, 240, 0.9)",
          borderRadius: "0.2rem",
          userSelect: "none",
        }}
        px="xs"
        py="0.1rem"
      >
        {props.name}
      </Text>
    </Html>
  ) : null;
}

/** Component containing the three.js object and children for a particular scene node. */
export function SceneNodeThreeObject(props: {
  name: string;
  parent: THREE.Object3D | null;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const makeObject = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.makeObject,
  );
  const cleanup = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.cleanup,
  );
  const unmountWhenInvisible = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.unmountWhenInvisible,
  );
  const [unmount, setUnmount] = React.useState(false);
  const clickable =
    viewer.useSceneTree((state) => state.nodeFromName[props.name]?.clickable) ??
    false;
  const [obj, setRef] = React.useState<THREE.Object3D | null>(null);

  const dragInfo = React.useRef({
    dragging: false,
    startClientX: 0,
    startClientY: 0,
  });

  // Create object + children.
  //
  // For not-fully-understood reasons, wrapping makeObject with useMemo() fixes
  // stability issues (eg breaking runtime errors) associated with
  // PivotControls.
  const objNode = React.useMemo(
    () => makeObject && makeObject(setRef),
    [makeObject],
  );
  const children =
    obj === null ? null : (
      <SceneNodeThreeChildren name={props.name} parent={obj} />
    );

  // Helper for transient visibility checks. Checks the .visible attribute of
  // both this object and ancestors.
  //
  // This is used for (1) suppressing click events and (2) unmounting when
  // unmountWhenInvisible is true. The latter is used for <Html /> components.
  function isDisplayed() {
    // We avoid checking obj.visible because obj may be unmounted when
    // unmountWhenInvisible=true.
    if (viewer.nodeAttributesFromName.current[props.name]?.visibility === false)
      return false;
    if (props.parent === null) return true;

    // Check visibility of parents + ancestors.
    let visible = props.parent.visible;
    if (visible) {
      props.parent.traverseAncestors((ancestor) => {
        visible = visible && ancestor.visible;
      });
    }
    return visible;
  }

  // Update attributes on a per-frame basis. Currently does redundant work,
  // although this shouldn't be a bottleneck.
  useFrame(() => {
    if (unmountWhenInvisible) {
      const displayed = isDisplayed();
      if (displayed && unmount) {
        setUnmount(false);
      }
      if (!displayed && !unmount) {
        setUnmount(true);
      }
    }

    if (obj === null) return;

    const nodeAttributes = viewer.nodeAttributesFromName.current[props.name];
    if (nodeAttributes === undefined) return;

    const visibility = nodeAttributes.visibility;
    if (visibility !== undefined) {
      obj.visible = visibility;
    }

    let changed = false;
    const wxyz = nodeAttributes.wxyz;
    if (wxyz !== undefined) {
      changed = true;
      obj.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
    }
    const position = nodeAttributes.position;
    if (position !== undefined) {
      changed = true;
      obj.position.set(position[0], position[1], position[2]);
    }

    // Update matrices if necessary. This is necessary for PivotControls.
    if (changed) {
      if (!obj.matrixAutoUpdate) obj.updateMatrix();
      if (!obj.matrixWorldAutoUpdate) obj.updateMatrixWorld();
    }
  });

  // Clean up when done.
  React.useEffect(() => cleanup);

  // Clicking logic.
  const sendClicksThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    50,
  );
  const [hovered, setHovered] = React.useState(false);
  useCursor(hovered);
  if (!clickable && hovered) setHovered(false);

  if (objNode === undefined || unmount) {
    return <>{children}</>;
  } else if (clickable) {
    return (
      <>
        <group
          // Instead of using onClick, we use onPointerDown/Move/Up to check mouse drag,
          // and only send a click if the mouse hasn't moved between the down and up events.
          //  - onPointerDown resets the click state (dragged = false)
          //  - onPointerMove, if triggered, sets dragged = true
          //  - onPointerUp, if triggered, sends a click if dragged = false.
          // Note: It would be cool to have dragged actions too...
          onPointerDown={(e) => {
            if (!isDisplayed()) return;
            e.stopPropagation();
            const state = dragInfo.current;
            state.startClientX = e.clientX;
            state.startClientY = e.clientY;
            state.dragging = false;
          }}
          onPointerMove={(e) => {
            if (!isDisplayed()) return;
            e.stopPropagation();
            const state = dragInfo.current;
            const deltaX = e.clientX - state.startClientX;
            const deltaY = e.clientY - state.startClientY;
            // Minimum motion.
            if (Math.abs(deltaX) <= 3 && Math.abs(deltaY) <= 3) return;
            state.dragging = true;
          }}
          onPointerUp={(e) => {
            if (!isDisplayed()) return;
            e.stopPropagation();
            const state = dragInfo.current;
            if (state.dragging) return;
            sendClicksThrottled({
              type: "SceneNodeClickMessage",
              name: props.name,
              // Note that the threejs up is +Y, but we expose a +Z up.
              ray_origin: [e.ray.origin.x, -e.ray.origin.z, e.ray.origin.y],
              ray_direction: [
                e.ray.direction.x,
                -e.ray.direction.z,
                e.ray.direction.y,
              ],
            });
          }}
          onPointerOver={(e) => {
            if (!isDisplayed()) return;
            e.stopPropagation();
            setHovered(true);
          }}
          onPointerOut={() => {
            if (!isDisplayed()) return;
            setHovered(false);
          }}
        >
          <Select enabled={hovered}>{objNode}</Select>
        </group>
        {children}
      </>
    );
  } else {
    return (
      <>
        {/* This <group /> does nothing, but switching between clickable vs not
        causes strange transform behavior without it. */}
        <group>{objNode}</group>
        {children}
      </>
    );
  }
}
