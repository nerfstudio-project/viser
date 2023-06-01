import { createPortal } from "@react-three/fiber";
import { useCursor } from "@react-three/drei";
import React from "react";
import * as THREE from "three";

import { CoordinateFrame } from "./ThreeAssets";

import { immerable } from "immer";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { ViewerContext } from ".";
import { makeThrottledMessageSender } from "./WebsocketInterface";
import { Select } from "@react-three/postprocessing";

export type MakeObject<T extends THREE.Object3D = THREE.Object3D> = (
  ref: React.Ref<T>
) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode<T extends THREE.Object3D = THREE.Object3D> {
  [immerable] = true;

  public children: string[];

  constructor(
    public name: string,
    public makeObject: MakeObject<T>,
    public cleanup?: () => void
  ) {
    this.children = [];
  }
}

interface SceneTreeState {
  nodeFromName: { [name: string]: undefined | SceneNode };
  // Assignable attributes are defined separately from the node itself. This
  // ensures that assignments can happen before a node is created.
  attributesFromName: {
    [name: string]:
      | undefined
      | {
          visibility?: boolean;
          wxyz?: THREE.Quaternion;
          position?: THREE.Vector3;
          clickable?: boolean;
        };
  };
}
export interface SceneTreeActions extends SceneTreeState {
  setVisibility(name: string, visible: boolean): void;
  setOrientation(name: string, wxyz: THREE.Quaternion): void;
  setPosition(name: string, position: THREE.Vector3): void;
  setClickable(name: string, clickable: boolean): void;
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
        attributesFromName: {},
        setVisibility: (name, visibility) =>
          set((state) => {
            state.attributesFromName[name] = {
              ...state.attributesFromName[name],
              visibility: visibility,
            };
          }),
        setOrientation: (name, wxyz) =>
          set((state) => {
            state.attributesFromName[name] = {
              ...state.attributesFromName[name],
              wxyz: wxyz,
            };
          }),
        setPosition: (name, position) =>
          set((state) => {
            state.attributesFromName[name] = {
              ...state.attributesFromName[name],

              position: position,
            };
          }),
        setClickable: (name, clickable) =>
          set((state) => {
            state.attributesFromName[name] = {
              ...state.attributesFromName[name],

              clickable: clickable,
            };
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
  parent: THREE.Object3D;
}) {
  const children = props.useSceneTree(
    (state) => state.nodeFromName[props.name]?.children
  );

  // Can't make children inside of the parent until the parent exists.
  if (children === undefined) return <></>;

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
    props.parent
  );
}

/** Component containing the three.js object and children for a particular scene node. */
export function SceneNodeThreeObject(props: {
  name: string;
  useSceneTree: UseSceneTree;
}) {
  const { makeObject, cleanup } = props.useSceneTree(
    (state) => state.nodeFromName[props.name]!
  );
  const { visibility, wxyz, position, clickable } = props.useSceneTree(
    (state) => state.attributesFromName[props.name] || {}
  );

  const [obj, setRef] = React.useState<THREE.Object3D | null>(null);

  const { objFromSceneNodeNameRef, websocketRef } =
    React.useContext(ViewerContext)!;

  React.useEffect(() => {
    if (obj === null) return;

    if (visibility !== undefined) obj.visible = visibility;
    if (wxyz !== undefined) obj.rotation.setFromQuaternion(wxyz);
    if (position !== undefined)
      obj.position.set(position.x, position.y, position.z);

    // Update matrices if necessary. This is necessary for PivotControls.
    if (!obj.matrixAutoUpdate) obj.updateMatrix();
    if (!obj.matrixWorldAutoUpdate) obj.updateMatrixWorld();
  }, [obj, visibility, wxyz, position]);

  React.useEffect(() => {
    if (obj === null) return;

    objFromSceneNodeNameRef.current[props.name] = obj;
    return () => {
      delete objFromSceneNodeNameRef.current[props.name];
      cleanup && cleanup();
    };
  }, [obj, cleanup]);

  const sendClicksThrottled = makeThrottledMessageSender(websocketRef, 50);
  const [hovered, setHovered] = React.useState(false);
  const objNode = React.useMemo(() => makeObject(setRef), [makeObject, setRef]);
  const children = obj !== null && (
    <SceneNodeThreeChildren
      name={props.name}
      useSceneTree={props.useSceneTree}
      parent={obj}
    />
  );
  useCursor(hovered);

  if (clickable)
    return (
      <>
        <group
          onClick={(e) => {
            e.stopPropagation();
            console.log("foo", props.name);
            sendClicksThrottled({
              type: "SceneNodeClickedMessage",
              name: props.name,
            });
          }}
          onPointerOver={() => {
            setHovered(true);
          }}
          onPointerOut={() => {
            setHovered(false);
          }}
        >
          <Select enabled={hovered}>{objNode}</Select>
        </group>
        {children}
      </>
    );
  else {
    hovered && setHovered(false);
    return (
      <>
        {objNode}
        {children}
      </>
    );
  }
}
