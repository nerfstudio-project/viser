import { useCursor } from "@react-three/drei";
import { createPortal, useFrame } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { CoordinateFrame } from "./ThreeAssets";

import { ViewerContext } from ".";
import { makeThrottledMessageSender } from "./WebsocketInterface";
import { Html } from "@react-three/drei";
import { Select } from "@react-three/postprocessing";
import { immerable } from "immer";
import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { Text } from "@mantine/core";

export type MakeObject<T extends THREE.Object3D = THREE.Object3D> = (
  ref: React.Ref<T>
) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode<T extends THREE.Object3D = THREE.Object3D> {
  [immerable] = true;

  public children: string[];
  public clickable: boolean;

  constructor(
    public name: string,
    public makeObject: MakeObject<T>,
    public cleanup?: () => void
  ) {
    this.children = [];
    this.clickable = false;
  }
}

interface SceneTreeState {
  nodeFromName: { [name: string]: undefined | SceneNode };
  // Putting this into SceneNode makes the scene tree table much harder to implement.
  labelVisibleFromName: { [name: string]: boolean };
}
export interface SceneTreeActions extends SceneTreeState {
  setClickable(name: string, clickable: boolean): void;
  addSceneNode(nodes: SceneNode): void;
  removeSceneNode(name: string): void;
  resetScene(): void;
  setLabelVisibility(name: string, labelVisibility: boolean): void;
}

// Create default scene tree state.
// By default, the y-axis is up. Let's rotate everything so Z is up instead.
const makeRoot: MakeObject<THREE.Group> = (ref) => (
  <group
    ref={ref}
    quaternion={new THREE.Quaternion().setFromEuler(
      new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0)
    )}
  />
);
const rootAxesTemplate: MakeObject<THREE.Group> = (ref) => (
  <CoordinateFrame ref={ref} />
);

const rootNodeTemplate = new SceneNode(
  "",
  makeRoot
) as SceneNode<THREE.Object3D>;

const rootAxesNode = new SceneNode(
  "/WorldAxes",
  rootAxesTemplate
) as SceneNode<THREE.Object3D>;
rootNodeTemplate.children.push("/WorldAxes");

/** Declare a scene state, and return a hook for accessing it. Note that we put
effort into avoiding a global state! */
export function useSceneTreeState() {
  return React.useState(() =>
    create(
      subscribeWithSelector(
        immer<SceneTreeState & SceneTreeActions>((set) => ({
          nodeFromName: { "": rootNodeTemplate, "/WorldAxes": rootAxesNode },
          labelVisibleFromName: {},
          setClickable: (name, clickable) =>
            set((state) => {
              const node = state.nodeFromName[name];
              if (node !== undefined) node.clickable = clickable;
            }),
          addSceneNode: (node) =>
            set((state) => {
              const existingNode = state.nodeFromName[node.name];
              if (existingNode) {
                // Node already exists.
                state.nodeFromName[node.name] = {
                  ...node,
                  children: existingNode.children,
                };
              } else {
                // Node doesn't exist yet!
                // TODO: this assumes the parent exists. We could probably merge this with addSceneNodeMakeParents.
                const parent_name = node.name.split("/").slice(0, -1).join("/");
                state.nodeFromName[node.name] = node;
                state.nodeFromName[parent_name]!.children.push(node.name);
              }
            }),
          removeSceneNode: (name) =>
            set((state) => {
              if (!(name in state.nodeFromName)) {
                console.log("Skipping scene node removal for " + name);
                return;
              }

              // Remove this scene node and all children.
              const removeNames: string[] = [];
              function findChildrenRecursive(name: string) {
                removeNames.push(name);
                state.nodeFromName[name]!.children.forEach(
                  findChildrenRecursive
                );
              }
              findChildrenRecursive(name);

              removeNames.forEach((removeName) => {
                delete state.nodeFromName[removeName];
              });

              // Remove node from parent's children list.
              const parent_name = name.split("/").slice(0, -1).join("/");
              state.nodeFromName[parent_name]!.children = state.nodeFromName[
                parent_name
              ]!.children.filter((child_name) => child_name !== name);
            }),
          resetScene: () =>
            set((state) => {
              // For scene resets: we need to retain the object references created for the root and world frame nodes.
              for (const key of Object.keys(state.nodeFromName)) {
                if (key !== "" && key !== "/WorldAxes")
                  delete state.nodeFromName[key];
              }
              state.nodeFromName[""]!.children = ["/WorldAxes"];
              state.nodeFromName["/WorldAxes"]!.children = [];
            }),
          setLabelVisibility: (name, labelVisibility) =>
            set((state) => {
              state.labelVisibleFromName[name] = labelVisibility;
            }),
        }))
      )
    )
  )[0];
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

function SceneNodeThreeChildren(props: {
  name: string;
  parent: THREE.Object3D;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const [children, setChildren] = React.useState<string[]>([]);

  // De-bounce updates to children.
  React.useEffect(() => {
    let readyToUpdate = true;

    let updateChildrenTimeout: NodeJS.Timeout | undefined = undefined;

    function updateChildren() {
      const newChildren =
        viewer.useSceneTree.getState().nodeFromName[props.name]?.children;
      if (newChildren === undefined || children == newChildren) {
        return;
      }
      if (readyToUpdate) {
        setChildren(newChildren!);
        readyToUpdate = false;
        updateChildrenTimeout = setTimeout(() => {
          readyToUpdate = true;
          updateChildren();
        }, 50);
      }
    }
    const unsubscribe = viewer.useSceneTree.subscribe(
      (state) => state.nodeFromName[props.name],
      updateChildren
    );
    updateChildren();

    return () => {
      clearTimeout(updateChildrenTimeout);
      unsubscribe();
    };
  }, [children]);

  // Create a group of children inside of the parent object.
  return createPortal(
    <group>
      {children.map((child_id) => {
        return <SceneNodeThreeObject key={child_id} name={child_id} />;
      })}
    </group>,
    props.parent
  );
}

/** Component for updating attributes of a scene node. */
function SceneNodeLabel(props: { name: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const labelVisible = viewer.useSceneTree(
    (state) => state.labelVisibleFromName[props.name]
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
export function SceneNodeThreeObject(props: { name: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const makeObject = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.makeObject
  );
  const cleanup = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.cleanup
  );
  const clickable =
    viewer.useSceneTree((state) => state.nodeFromName[props.name]?.clickable) ??
    false;
  const [obj, setRef] = React.useState<THREE.Object3D | null>(null);

  // Create object + children.
  const objNode = React.useMemo(
    () => makeObject && makeObject(setRef),
    [setRef, makeObject]
  );
  const children = React.useMemo(
    () =>
      obj === null ? null : (
        <SceneNodeThreeChildren name={props.name} parent={obj} />
      ),
    [props.name, obj]
  );

  // Update attributes on a per-frame basis. Currently does redundant work,
  // although this shouldn't be a bottleneck.
  useFrame(() => {
    if (obj === null) return;

    const nodeAttributes = viewer.nodeAttributesFromName.current[props.name];
    const wxyz = nodeAttributes?.wxyz;
    const position = nodeAttributes?.position;
    const visibility = nodeAttributes?.visibility;

    let changed = false;
    if (visibility !== undefined) obj.visible = visibility;
    if (wxyz !== undefined) {
      changed = true;
      obj.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
    }
    if (position !== undefined) {
      changed = true;
      obj.position.set(position[0], position[1], position[2]);
    }

    // Update matrices if necessary. This is necessary for PivotControls.
    if (changed && !obj.matrixAutoUpdate) obj.updateMatrix();
    if (changed && !obj.matrixWorldAutoUpdate) obj.updateMatrixWorld();
  });

  // Clean up when done.
  React.useEffect(() => cleanup);

  // Clicking logic.
  const sendClicksThrottled = makeThrottledMessageSender(
    viewer.websocketRef,
    50
  );
  const [hovered, setHovered] = React.useState(false);
  useCursor(hovered);
  if (!clickable && hovered) setHovered(false);

  // Helper for checking transient visibility checks.
  function isVisible() {
    const nodeAttributes = viewer.nodeAttributesFromName.current[props.name];
    return nodeAttributes?.visibility ?? false;
  }

  if (clickable) {
    return (
      <>
        <group
          onClick={
            !clickable
              ? undefined
              : (e) => {
                  if (!isVisible()) return;
                  e.stopPropagation();
                  sendClicksThrottled({
                    type: "SceneNodeClickedMessage",
                    name: props.name,
                  });
                }
          }
          onPointerOver={
            !clickable
              ? undefined
              : (e) => {
                  if (!isVisible()) return;
                  e.stopPropagation();
                  setHovered(true);
                }
          }
          onPointerOut={
            !clickable
              ? undefined
              : () => {
                  if (!isVisible()) return;
                  setHovered(false);
                }
          }
        >
          <Select enabled={hovered}>{objNode}</Select>
        </group>
        <SceneNodeLabel name={props.name} />
        {children}
      </>
    );
  } else {
    return (
      <>
        {objNode}
        <SceneNodeLabel name={props.name} />
        {children}
      </>
    );
  }
}
