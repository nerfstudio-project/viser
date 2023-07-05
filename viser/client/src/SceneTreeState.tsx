import React from "react";
import { MakeObject, SceneNode } from "./SceneTree";
import { CoordinateFrame } from "./ThreeAssets";
import * as THREE from "three";
import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

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
