import React from "react";
import { MakeObject, SceneNode } from "./SceneTree";
import { CoordinateFrame } from "./ThreeAssets";
import * as THREE from "three";
import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { EnvironmentMapMessage } from "./WebsocketMessages";

interface SceneTreeState {
  nodeFromName: { [name: string]: undefined | SceneNode };
  // Putting this into SceneNode makes the scene tree table much harder to implement.
  labelVisibleFromName: { [name: string]: boolean };
  enableDefaultLights: boolean;
  environmentMap: EnvironmentMapMessage;
}
export interface SceneTreeActions extends SceneTreeState {
  setClickable(name: string, clickable: boolean): void;
  addSceneNode(nodes: SceneNode): void;
  removeSceneNode(name: string): void;
  resetScene(): void;
  setLabelVisibility(name: string, labelVisibility: boolean): void;
}

// Create default scene tree state.
const rootAxesTemplate: MakeObject<THREE.Group> = (ref) => (
  <CoordinateFrame ref={ref} />
);

const rootNodeTemplate = new SceneNode<THREE.Group>("", (ref) => (
  <group ref={ref} />
)) as SceneNode<THREE.Object3D>;

const rootAxesNode = new SceneNode(
  "/WorldAxes",
  rootAxesTemplate,
) as SceneNode<THREE.Object3D>;
rootNodeTemplate.children.push("/WorldAxes");

/** Declare a scene state, and return a hook for accessing it. Note that we put
effort into avoiding a global state! */
export function useSceneTreeState(
  nodeRefFromName: React.MutableRefObject<{
    [name: string]: undefined | THREE.Object3D;
  }>,
) {
  return React.useState(() =>
    create(
      subscribeWithSelector(
        immer<SceneTreeState & SceneTreeActions>((set) => ({
          nodeFromName: { "": rootNodeTemplate, "/WorldAxes": rootAxesNode },
          labelVisibleFromName: {},
          enableDefaultLights: true,
          environmentMap: {
            type: "EnvironmentMapMessage",
            hdri: "city",
            background: false,
            background_blurriness: 0,
            background_intensity: 1,
            background_rotation: [0, 0, 0],
            environment_intensity: 1,
            environment_rotation: [0, 0, 0],
          },
          setClickable: (name, clickable) =>
            set((state) => {
              const node = state.nodeFromName[name];
              if (node !== undefined) node.clickable = clickable;
            }),
          addSceneNode: (node) =>
            set((state) => {
              const existingNode = state.nodeFromName[node.name];
              if (existingNode !== undefined) {
                // Node already exists.
                delete nodeRefFromName.current[node.name];
                existingNode.cleanup && existingNode.cleanup(); // Free resources.
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
                  findChildrenRecursive,
                );
              }
              findChildrenRecursive(name);

              removeNames.forEach((removeName) => {
                const node = state.nodeFromName[removeName]!;
                node.cleanup && node.cleanup(); // Free resources.
                delete state.nodeFromName[removeName];
                delete nodeRefFromName.current[removeName];
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
              Object.values(state.nodeFromName).forEach((node) => {
                // Free resources.
                if (node === undefined || node.cleanup === undefined) return;
                node.cleanup();
              });
              state.nodeFromName[""] = rootNodeTemplate;
              state.nodeFromName["/WorldAxes"] = rootAxesNode;
            }),
          setLabelVisibility: (name, labelVisibility) =>
            set((state) => {
              state.labelVisibleFromName[name] = labelVisibility;
            }),
        })),
      ),
    ),
  )[0];
}
