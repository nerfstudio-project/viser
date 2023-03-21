import React from "react";
import create from "zustand";
import { immer } from "zustand/middleware/immer";

interface GuiState {
  label: string;
  server: string;
  websocketConnected: boolean;
  guiConfigFromName: { [key: string]: any };
}

interface GuiActions {
  setLabel: (label: string) => void;
  setServer: (server: string) => void;
  setWebsocketConnected: (connected: boolean) => void;
  addGui: (name: string, value: any) => void;
  resetGui: () => void;
}

const cleanGuiState = {
  label: "",
  server: "ws://localhost:8080",
  websocketConnected: false,
  guiConfigFromName: {},
} as GuiState;

export function useGuiState() {
  return React.useState(() =>
    create(
      immer<GuiState & GuiActions>((set) => ({
        ...cleanGuiState,
        setLabel: (label) =>
          set((state) => {
            state.label = label;
          }),
        setServer: (server) =>
          set((state) => {
            state.server = server;
          }),
        setWebsocketConnected: (connected) =>
          set((state) => {
            state.websocketConnected = connected;
          }),
        addGui: (name, value) =>
          set((state) => {
            state.guiConfigFromName[name] = value;
          }),
        resetGui: () =>
          set((state) => {
            state.guiConfigFromName = {};
          }),
      }))
    )
  )[0];
}

/** Type corresponding to a zustand-style useGuiState hook. */
export type UseGui = ReturnType<typeof useGuiState>;
