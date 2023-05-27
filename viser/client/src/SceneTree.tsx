import { createPortal } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { CoordinateFrame } from "./ThreeAssets";

import { immerable } from "immer";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export type MakeObject<T extends THREE.Object3D = THREE.Object3D> = (
  ref: React.RefObject<T>
) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode<T extends THREE.Object3D = THREE.Object3D> {
  [immerable] = true;

  public children: string[];

  constructor(
    public name: string,
    public makeObject: MakeObject<T>,
    public cleanup?: () => void,
    public visibility: boolean = true,
    public wxyz?: THREE.Quaternion,
    public position?: THREE.Vector3,
    public obj?: THREE.Object3D
  ) {
    this.children = [];
  }
}

interface SceneTreeState {
  nodeFromName: { [key: string]: SceneNode | undefined };
}
export interface SceneTreeActions extends SceneTreeState {
  setObj(name: string, obj: THREE.Object3D): void;
  setVisibility(name: string, visible: boolean): void;
  setOrientation(name: string, wxyz: THREE.Quaternion): void;
  setPosition(name: string, position: THREE.Vector3): void;
  clearObj(name: string): void;
  addSceneNode(nodes: SceneNode): void;
  removeSceneNode(name: string): void;
  resetScene(): void;
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
      immer<SceneTreeState & SceneTreeActions>((set) => ({
        nodeFromName: { "": rootNodeTemplate, "/WorldAxes": rootAxesNode },
        setObj: (name, obj) =>
          set((state) => {
            const node = state.nodeFromName[name];
            if (node) node.obj = obj;
          }),
        setVisibility: (name, visibility) =>
          set((state) => {
            const node = state.nodeFromName[name];
            if (node) node.visibility = visibility;
          }),
        setOrientation: (name, wxyz) =>
          set((state) => {
            const node = state.nodeFromName[name];
            if (node) node.wxyz = wxyz;
          }),
        setPosition: (name, position) =>
          set((state) => {
            const node = state.nodeFromName[name];
            if (node) node.position = position;
          }),
        clearObj: (name) =>
          set((state) => {
            const node = state.nodeFromName[name];
            if (node) node.obj = undefined;
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
              state.nodeFromName[name]!.children.forEach(findChildrenRecursive);
            }
            findChildrenRecursive(name);

            removeNames.forEach((removeName) => {
              delete state.nodeFromName[removeName];
            });
          }),
        resetScene: () =>
          set((state) => {
            // For scene resets: we need to retain the object references created for the root and world frame nodes.
            for (const key of Object.keys(state.nodeFromName)) {
              if (key !== "" && key !== "/WorldAxes")
                delete state.nodeFromName[key];
            }
            state.nodeFromName[""]!.children = [];
            state.nodeFromName["/WorldAxes"]!.children = [];
          }),
      }))
    )
  )[0];
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

function SceneNodeThreeChildren(props: {
  name: string;
  useSceneTree: UseSceneTree;
}) {
  const [children, parentObj] = props.useSceneTree((state) => {
    const node = state.nodeFromName[props.name];
    return [node?.children, node?.obj];
  });

  // Can't make children inside of the parent until the parent exists.
  if (children == undefined || parentObj === undefined) return <></>;

  // Create a group of children inside of the parent object.
  return createPortal(
    <group>
      {children.map((child_id) => {
        return (
          <SceneNodeThreeObject
            key={child_id}
            name={child_id}
            useSceneTree={props.useSceneTree}
          />
        );
      })}
    </group>,
    parentObj
  );
}

/** Component containing the three.js object and children for a particular scene node. */
export function SceneNodeThreeObject(props: {
  name: string;
  useSceneTree: UseSceneTree;
}) {
  const [makeObject, cleanup] = props.useSceneTree((state) => {
    const node = state.nodeFromName[props.name];
    return [node?.makeObject, node?.cleanup];
  });
  const setObj = props.useSceneTree((state) => state.setObj);
  const clearObj = props.useSceneTree((state) => state.clearObj);
  const ref = React.useRef<THREE.Object3D>(null);

  React.useEffect(() => {
    setObj(props.name, ref.current!);
    return () => {
      clearObj(props.name);
      cleanup && cleanup();
    };
  }, [clearObj, cleanup]);

  if (makeObject === undefined) return <></>;
  return (
    <>
      {makeObject(ref)}
      <SceneNodeUpdater
        name={props.name}
        objRef={ref}
        useSceneTree={props.useSceneTree}
      />
      <SceneNodeThreeChildren
        name={props.name}
        useSceneTree={props.useSceneTree}
      />
    </>
  );
}

/** Shove visibility updates into a separate components so the main object
 * component doesn't need to be repeatedly re-rendered.*/
function SceneNodeUpdater(props: {
  name: string;
  objRef: React.RefObject<THREE.Object3D>;
  useSceneTree: UseSceneTree;
}) {
  const [visible, wxyz, position] = props.useSceneTree((state) => {
    const node = state.nodeFromName[props.name];
    return [node?.visibility, node?.wxyz, node?.position];
  });
  React.useEffect(() => {
    if (props.objRef.current === null) return;
    const obj = props.objRef.current;
    if (visible !== undefined) obj.visible = visible;

    wxyz && obj.rotation && obj.rotation.setFromQuaternion(wxyz);
    position &&
      obj.position &&
      obj.position.set(position.x, position.y, position.z);

    // Update matrices if necessary. This is necessary for PivotControls.
    if (!obj.matrixAutoUpdate) obj.updateMatrix();
    if (!obj.matrixWorldAutoUpdate) obj.updateMatrixWorld();
  }, [props, visible, wxyz, position]);
  return <></>;
}
