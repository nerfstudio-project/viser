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
import { Html } from "@react-three/drei";
import { Box, Text } from "@mantine/core";

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
          labelVisibility?: boolean;
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
      immer<SceneTreeState & SceneTreeActions>((set) => ({
        nodeFromName: { "": rootNodeTemplate, "/WorldAxes": rootAxesNode },
        attributesFromName: {
          "": { visibility: true },
          "/WorldAxes": { visibility: true },
        },
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
            state.attributesFromName[node.name] = {
              visibility: true,
              ...state.attributesFromName[node.name],
            };
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
            state.nodeFromName[""]!.children = ["/WorldAxes"];
            state.nodeFromName["/WorldAxes"]!.children = [];
          }),
        setLabelVisibility: (name, labelVisibility) =>
          set((state) => {
            state.attributesFromName[name] = {
              ...state.attributesFromName[name],
              labelVisibility: labelVisibility,
            };
          }),
      }))
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
  const children = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.children
  );

  // Can't make children inside of the parent until the parent exists.
  if (children === undefined) return <></>;

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

/** Component for updating attributes of a scene node.
 *
 * This is intentionally factored into its own component to reduce re-renders
 * of the full scene node + children.*/
function SceneNodeAttributeHandler(props: {
  obj: THREE.Object3D | null;
  name: string;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const { visibility, wxyz, position, labelVisibility } = viewer.useSceneTree(
    (state) => state.attributesFromName[props.name] || {}
  );

  React.useEffect(() => {
    if (props.obj === null) return;

    if (visibility !== undefined) props.obj.visible = visibility;
    if (wxyz !== undefined) props.obj.rotation.setFromQuaternion(wxyz);
    if (position !== undefined)
      props.obj.position.set(position.x, position.y, position.z);

    // Update matrices if necessary. This is necessary for PivotControls.
    if (!props.obj.matrixAutoUpdate) props.obj.updateMatrix();
    if (!props.obj.matrixWorldAutoUpdate) props.obj.updateMatrixWorld();
  }, [visibility, wxyz, position, props.obj]);

  return <SceneNodeLabel visible={labelVisibility} text={props.name} />;
}

/** Component containing the three.js object and children for a particular scene node. */
export function SceneNodeThreeObject(props: { name: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const { makeObject, cleanup } = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]!
  );

  const [obj, setRef] = React.useState<THREE.Object3D | null>(null);
  const { visibility, clickable } = viewer.useSceneTree(
    (state) => state.attributesFromName[props.name] || {}
  );

  // Hover state for clickable nodes.
  const [hovered, setHovered] = React.useState(false);
  useCursor(hovered);

  // Create object + children.
  const objNode = React.useMemo(() => makeObject(setRef), [setRef, makeObject]);
  const children = React.useMemo(
    () =>
      obj === null ? null : (
        <SceneNodeThreeChildren name={props.name} parent={obj} />
      ),
    [props.name, obj]
  );

  // Clean up when done.
  React.useEffect(() => cleanup);

  if (visibility && clickable) {
    // Clickable scene nodes. We don't include children.
    const sendClicksThrottled = makeThrottledMessageSender(
      viewer.websocketRef,
      50
    );
    return (
      <>
        <group
          onClick={(e) => {
            e.stopPropagation();
            sendClicksThrottled({
              type: "SceneNodeClickedMessage",
              name: props.name,
            });
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
          }}
          onPointerOut={() => {
            setHovered(false);
          }}
        >
          <Select enabled={hovered}>{objNode}</Select>
        </group>
        <SceneNodeAttributeHandler obj={obj} name={props.name} />
        {children}
      </>
    );
  } else {
    // Not clickable => not hovered!
    hovered && setHovered(false);
    return (
      <>
        {objNode}
        <SceneNodeAttributeHandler obj={obj} name={props.name} />
        {children}
      </>
    );
  }
}

type SceneNodeLabelProps = {
  text: string;
  visible?: boolean;
};

export function SceneNodeLabel({ text, visible }: SceneNodeLabelProps) {
  if (!visible) {
    return null;
  }
  return (
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
        {text}
      </Text>
    </Html>
  );
}
