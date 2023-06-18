import React from "react";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import {
  ThemeConfigurationMessage,
  GuiAddButtonMessage,
  GuiAddCheckboxMessage,
  GuiAddNumberMessage,
  GuiAddRgbMessage,
  GuiAddRgbaMessage,
  GuiAddSliderMessage,
  Message,
  GuiAddTextMessage,
} from "../WebsocketMessages";

export type GuiConfig =
  | GuiAddButtonMessage
  | GuiAddCheckboxMessage
  | GuiAddNumberMessage
  | GuiAddRgbMessage
  | GuiAddRgbaMessage
  | GuiAddSliderMessage
  | GuiAddTextMessage;

export function isGuiConfig(message: Message): message is GuiConfig {
  return message.type.startsWith("GuiAdd");
}

interface GuiState {
  theme: ThemeConfigurationMessage;
  label: string;
  server: string;
  websocketConnected: boolean;
  backgroundAvailable: boolean;
  guiConfigFromId: { [id: string]: GuiConfig };
  guiValueFromId: { [id: string]: any };
}

interface GuiActions {
  setTheme: (theme: ThemeConfigurationMessage) => void;
  addGui: (config: GuiConfig) => void;
  setGuiValue: (id: string, value: any) => void;
  removeGui: (name: string) => void;
  resetGui: () => void;
}

const cleanGuiState: GuiState = {
  theme: {
    type: "ThemeConfigurationMessage",
    titlebar_content: null,
    fixed_sidebar: false,
  },
  label: "",
  server: "ws://localhost:8080", // Currently this will always be overridden.
  websocketConnected: false,
  backgroundAvailable: false,
  guiConfigFromId: {},
  guiValueFromId: {},
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
        addGui: (guiConfig) =>
          set((state) => {
            state.guiConfigFromId[guiConfig.id] = guiConfig;
          }),
        setGuiValue: (id, value) =>
          set((state) => {
            state.guiValueFromId[id] = value;
          }),
        removeGui: (name) =>
          set((state) => {
            delete state.guiConfigFromId[name];
          }),
        resetGui: () =>
          set((state) => {
            state.guiConfigFromId = {};
          }),
      }))
    )
  )[0];
}

/** Type corresponding to a zustand-style useGuiState hook. */
export type UseGui = ReturnType<typeof useGuiState>;
