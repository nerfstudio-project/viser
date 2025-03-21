import React from "react";
import * as THREE from "three";
import { SceneNodeMessage } from "./WebsocketMessages";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { EnvironmentMapMessage } from "./WebsocketMessages";

export type SceneNode = {
  message: SceneNodeMessage;
  children: string[];
  clickable: boolean;
};

type SceneTreeState = {
  nodeFromName: { [name: string]: SceneNode | undefined };
  labelVisibleFromName: { [name: string]: boolean };
  enableDefaultLights: boolean;
  enableDefaultLightsShadows: boolean;
  environmentMap: EnvironmentMapMessage;
};

type SceneTreeActions = {
  setClickable(name: string, clickable: boolean): void;
  addSceneNode(message: SceneNodeMessage): void;
  removeSceneNode(name: string): void;
  updateSceneNode(name: string, updates: { [key: string]: any }): void;
  resetScene(): void;
  setLabelVisibility(name: string, labelVisibility: boolean): void;
};

// Pre-defined scene nodes.
export const rootNodeTemplate: SceneNode = {
  message: {
    type: "FrameMessage",
    name: "",
    props: {
      show_axes: false,
      axes_length: 0.5,
      axes_radius: 0.0125,
      origin_radius: 0.025,
      origin_color: [236, 236, 0],
    },
  },
  children: ["/WorldAxes"],
  clickable: false,
};
const worldAxesNodeTemplate: SceneNode = {
  message: {
    type: "FrameMessage",
    name: "/WorldAxes",
    props: {
      show_axes: true,
      axes_length: 0.5,
      axes_radius: 0.0125,
      origin_radius: 0.025,
      origin_color: [236, 236, 0],
    },
  },
  children: [],
  clickable: false,
};

/** Declare a scene state, and return a hook for accessing it. Note that we put
effort into avoiding a global state! */
export function useSceneTreeState(
  nodeRefFromName: React.MutableRefObject<{
    [name: string]: undefined | THREE.Object3D;
  }>,
) {
  return React.useState(() =>
    create(
      immer<SceneTreeState & SceneTreeActions>((set) => ({
        nodeFromName: {
          "": rootNodeTemplate,
          "/WorldAxes": worldAxesNodeTemplate,
        },
        labelVisibleFromName: {},
        enableDefaultLights: true,
        enableDefaultLightsShadows: false,
        environmentMap: {
          type: "EnvironmentMapMessage",
          hdri: "city",
          background: false,
          background_blurriness: 0,
          background_intensity: 1.0,
          background_wxyz: [1, 0, 0, 0],
          environment_intensity: 1.0,
          environment_wxyz: [1, 0, 0, 0],
        },
        setClickable: (name, clickable) =>
          set((state) => {
            const node = state.nodeFromName[name];
            if (node !== undefined) node.clickable = clickable;
          }),
        addSceneNode: (message) =>
          set((state) => {
            const existingNode = state.nodeFromName[message.name];
            if (existingNode !== undefined) {
              // Node already exists.
              delete nodeRefFromName.current[message.name];
              state.nodeFromName[message.name] = {
                ...existingNode,
                message: message,
              };
            } else {
              // Node doesn't exist yet!
              const parent_name = message.name
                .split("/")
                .slice(0, -1)
                .join("/");
              state.nodeFromName[message.name] = {
                message: message,
                children: [],
                clickable: false,
              };
              state.nodeFromName[parent_name]!.children.push(message.name);
            }
          }),
        removeSceneNode: (name) =>
          set((state) => {
            // Remove this scene node and all children.
            const removeNames: string[] = [];
            function findChildrenRecursive(name: string) {
              removeNames.push(name);
              state.nodeFromName[name]!.children.forEach(findChildrenRecursive);
            }
            findChildrenRecursive(name);

            removeNames.forEach((removeName) => {
              delete state.nodeFromName[removeName];
              delete nodeRefFromName.current[removeName];
            });

            // Remove node from parent's children list.
            const parent_name = name.split("/").slice(0, -1).join("/");
            state.nodeFromName[parent_name]!.children = state.nodeFromName[
              parent_name
            ]!.children.filter((child_name) => child_name !== name);
          }),
        updateSceneNode: (name, updates) =>
          set((state) => {
            if (state.nodeFromName[name] === undefined) {
              console.error(
                `Attempted to update non-existent node ${name} with updates:`,
                updates,
              );
              return;
            }
            state.nodeFromName[name]!.message.props = {
              ...state.nodeFromName[name]!.message.props,
              ...updates,
            };
          }),
        resetScene: () =>
          set((state) => {
            // For scene resets: we need to retain the object references created for the root and world frame nodes.
            for (const key of Object.keys(state.nodeFromName)) {
              if (key !== "" && key !== "/WorldAxes")
                delete state.nodeFromName[key];
            }
            state.nodeFromName[""] = rootNodeTemplate;
            state.nodeFromName["/WorldAxes"] = worldAxesNodeTemplate;
          }),
        setLabelVisibility: (name, labelVisibility) =>
          set((state) => {
            state.labelVisibleFromName[name] = labelVisibility;
          }),
      })),
    ),
  )[0];
}
