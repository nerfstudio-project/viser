import React from "react";
import { create, StoreApi, UseBoundStore } from "zustand";
import { EnvironmentMapMessage } from "./WebsocketMessages";

type EnvironmentState = {
  enableDefaultLights: boolean;
  enableDefaultLightsShadows: boolean;
  environmentMap: EnvironmentMapMessage;
};

/** Declare an environment state, and return a hook for accessing it. Note that we put
effort into avoiding a global state! */
export function useEnvironmentState() {
  return React.useState(() =>
    create<EnvironmentState>(() => ({
      enableDefaultLights: true,
      enableDefaultLightsShadows: true,
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
    })),
  )[0];
}
