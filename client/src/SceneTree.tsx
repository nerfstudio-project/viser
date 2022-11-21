import { createPortal } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { CoordinateFrame } from "./ThreeAssets";

import { immerable } from "immer";
import create from "zustand";
import { immer } from "zustand/middleware/immer";

export type NodeIdType = number;

// The covariance/contravariance rules are too complicated here, so we just
// type the reference with any.
export type MakeObject = (ref: React.RefObject<any>) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode {
  [immerable] = true;

  public children: NodeIdType[];

  constructor(public name: string, public make_object: MakeObject) {
    this.children = [];
  }
}

interface SceneTreeState {
  nodeCounter: number;
  nodeFromId: { [key: NodeIdType]: SceneNode };
  idFromName: { [key: string]: NodeIdType };
  objFromId: { [key: NodeIdType]: THREE.Object3D };
}
export interface SceneTree extends SceneTreeState {
  setObj(id: NodeIdType, obj: THREE.Object3D): void;
  clearObj(id: NodeIdType): void;
  addSceneNode(nodes: SceneNode): void;
  removeSceneNode(name: string): void;
  resetScene(): void;
}

// Create default scene tree state.
// By default, the y-axis is up. Let's rotate everything so Z is up instead.
// Are there return types with TypeScript?
const rootFrameTemplate: MakeObject = (ref) => (
  <CoordinateFrame
    ref={ref}
    scale={5.0}
    quaternion={new THREE.Quaternion().setFromEuler(
      new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0)
    )}
  />
);
const rootNodeTemplate = new SceneNode("", rootFrameTemplate);
const cleanSceneTreeState = {
  nodeCounter: 1,
  nodeFromId: { 0: rootNodeTemplate },
  idFromName: { "": 0 },
  objFromId: {},
} as SceneTreeState;

/** Declare a scene state, and return a hook for accessing it. Note that we put
effort into avoiding a global state! */
export function useSceneTreeState() {
  return React.useState(() =>
    create(
      immer<SceneTree>((set) => ({
        ...cleanSceneTreeState,
        setObj: (id, obj) =>
          set((state) => {
            state.objFromId[id] = obj;
          }),
        clearObj: (id) =>
          set((state) => {
            delete state.objFromId[id];
          }),
        addSceneNode: (node) =>
          set((state) => {
            if (state.idFromName[node.name] !== undefined) {
              console.log("Updating node:", node.name);
              const id = state.idFromName[node.name];
              state.nodeFromId[id] = {
                ...node,
                children: state.nodeFromId[id].children,
              };
            } else {
              console.log("Creating node:", node.name);
              const id = state.nodeCounter;

              const parent_name = node.name.split("/").slice(0, -1).join("/");
              const parent_id = state.idFromName[parent_name];
              state.nodeFromId[id] = node;
              state.nodeFromId[parent_id].children.push(id);
              state.idFromName[node.name] = id;
              state.nodeCounter++;
            }
          }),
        removeSceneNode: (name) =>
          set((state) => {
            // Remove node from parent's children list.
            const parent_name = name.split("/").slice(0, -1).join("/");

            const remove_id = state.idFromName[name];
            const parent_id = state.idFromName[parent_name];
            state.nodeFromId[parent_id].children = state.nodeFromId[
              parent_id
            ].children.filter((id) => id !== remove_id);

            // If we want to remove "/tree", we should remove all of "/tree", "/tree/trunk", "/tree/branch", etc.
            const remove_names = Object.keys(state.idFromName).filter((n) =>
              n.startsWith(name)
            );
            remove_names.forEach((remove_name) => {
              const id = state.idFromName[remove_name];
              delete state.nodeFromId[id];
              delete state.idFromName[remove_name];
            });
          }),
        resetScene: () =>
          set((state) => {
            Object.assign(state, cleanSceneTreeState);
          }),
      }))
    )
  )[0];
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

// How common is typing via interfaces like this?
interface SceneNodeThreeChildrenProps {
  id: NodeIdType;
  useSceneTree: UseSceneTree;
}
function SceneNodeThreeChildren(props: SceneNodeThreeChildrenProps) {
  const children = props.useSceneTree(
    (state) => state.nodeFromId[props.id].children
  );
  const parentObj = props.useSceneTree((state) => state.objFromId[props.id]);

  // Create a group of children inside of the parent object.
  return (
    parentObj &&
    createPortal(
      <group>
        {children.map((child_id) => {
          return (
            <SceneNodeThreeObject
              key={child_id}
              id={child_id}
              useSceneTree={props.useSceneTree}
            />
          );
        })}
      </group>,
      parentObj
    )
  );
}

interface SceneNodeThreeObjectProps {
  id: NodeIdType;
  useSceneTree: UseSceneTree; // How does "UseSceneTree" work here?
}

/** Component containing the three.js object and children for a particular scene node. */
export function SceneNodeThreeObject(props: SceneNodeThreeObjectProps) {
  const sceneNode = props.useSceneTree((state) => state.nodeFromId[props.id]);
  const setObj = props.useSceneTree((state) => state.setObj);
  const clearObj = props.useSceneTree((state) => state.clearObj);
  const ref = React.useRef<THREE.Object3D>();

  React.useEffect(() => {
    setObj(props.id, ref.current!);
    return () => clearObj(props.id);
  });

  return (
    <>
      {sceneNode.make_object(ref)}
      <SceneNodeThreeChildren id={props.id} useSceneTree={props.useSceneTree} />
    </>
  );
}
