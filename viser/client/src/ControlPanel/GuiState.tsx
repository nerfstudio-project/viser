import React from "react";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { Message } from "../WebsocketMessages";

// Individual message types are not exported, so we extract just one type from
// the message union via an intersection.
type ThemeConfigurationMessage = Message & {
  type: "ThemeConfigurationMessage";
};

interface GuiConfig {
  levaConf: any;
  folderLabels: string[];
  visible: boolean;
}

interface GuiState {
  theme: ThemeConfigurationMessage;
  label: string;
  server: string;
  websocketConnected: boolean;
  backgroundAvailable: boolean;
  guiNames: string[]; // Used to retain input ordering.
  guiConfigFromName: { [key: string]: GuiConfig };
  guiSetQueue: { [key: string]: any };
}

interface GuiActions {
  setTheme: (theme: ThemeConfigurationMessage) => void;
  addGui: (name: string, config: GuiConfig) => void;
  removeGui: (name: string) => void;
  resetGui: () => void;
  guiSet: (name: string, value: any) => void;
  applyGuiSetQueue: (apply: (name: string, value: any) => void) => void;
}

const cleanGuiState: GuiState = {
  theme: {
    type: "ThemeConfigurationMessage",
    canvas_background_color: 0xffffff,
    titlebar_content: null
  },
  label: "",
  server: "ws://localhost:8080", // Currently this will always be overridden.
  websocketConnected: false,
  backgroundAvailable: false,
  guiNames: [],
  guiConfigFromName: {},
  guiSetQueue: {},
};

export function useGuiState(initialServer: string) {
  return React.useState(() =>
    create(
      immer<GuiState & GuiActions>((set) => ({
        ...cleanGuiState,
        server: initialServer,
        setTheme: (theme) =>
          set((state) => {
            state.theme = theme;
          }),
        addGui: (name, guiConfig) =>
          set((state) => {
            state.guiNames.push(name);
            state.guiConfigFromName[name] = guiConfig;
          }),
        removeGui: (name) =>
          set((state) => {
            state.guiNames.splice(state.guiNames.indexOf(name), 1);
            delete state.guiConfigFromName[name];
          }),
        resetGui: () =>
          set((state) => {
            state.guiNames = [];
            state.guiConfigFromName = {};
          }),
        guiSet: (name, value) =>
          set((state) => {
            state.guiSetQueue[name] = value;
          }),
        applyGuiSetQueue: (apply) =>
          set((state) => {
            for (const [key, value] of Object.entries(state.guiSetQueue)) {
              // Linear runtime here could be improved.
              if (state.guiNames.includes(key)) {
                delete state.guiSetQueue[key];
                apply(key, value);
              }
            }
          }),
      }))
    )
  )[0];
}

/** Type corresponding to a zustand-style useGuiState hook. */
export type UseGui = ReturnType<typeof useGuiState>;
