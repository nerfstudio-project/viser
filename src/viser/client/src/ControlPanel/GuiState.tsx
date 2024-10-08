import React from "react";
import { create } from "zustand";
import { ColorTranslator } from "colortranslator";

import { immer } from "zustand/middleware/immer";
import {
  GuiComponentMessage,
  GuiModalMessage,
  ThemeConfigurationMessage,
} from "../WebsocketMessages";

interface GuiState {
  theme: ThemeConfigurationMessage;
  label: string;
  server: string;
  shareUrl: string | null;
  websocketConnected: boolean;
  backgroundAvailable: boolean;
  guiIdSetFromContainerId: {
    [containerId: string]: { [id: string]: true } | undefined;
  };
  modals: GuiModalMessage[];
  guiOrderFromId: { [id: string]: number };
  guiConfigFromId: { [id: string]: GuiComponentMessage };
  uploadsInProgress: {
    [id: string]: {
      notificationId: string;
      uploadedBytes: number;
      totalBytes: number;
      filename: string;
    };
  };
}

interface GuiActions {
  setTheme: (theme: ThemeConfigurationMessage) => void;
  setShareUrl: (share_url: string | null) => void;
  addGui: (config: GuiComponentMessage) => void;
  addModal: (config: GuiModalMessage) => void;
  removeModal: (id: string) => void;
  updateGuiProps: (id: string, updates: { [key: string]: any }) => void;
  removeGui: (id: string) => void;
  resetGui: () => void;
  updateUploadState: (
    state: (
      | { uploadedBytes: number; totalBytes: number }
      | GuiState["uploadsInProgress"][string]
    ) & { componentId: string },
  ) => void;
}

const cleanGuiState: GuiState = {
  theme: {
    type: "ThemeConfigurationMessage",
    titlebar_content: null,
    control_layout: "floating",
    control_width: "medium",
    dark_mode: false,
    show_logo: true,
    show_share_button: true,
    colors: null,
  },
  label: "",
  server: "ws://localhost:8080", // Currently this will always be overridden.
  shareUrl: null,
  websocketConnected: false,
  backgroundAvailable: false,
  guiIdSetFromContainerId: {},
  modals: [],
  guiOrderFromId: {},
  guiConfigFromId: {},
  uploadsInProgress: {},
};

export function computeRelativeLuminance(color: string) {
  const colorTrans = new ColorTranslator(color);

  // Coefficients are from:
  // https://en.wikipedia.org/wiki/Relative_luminance#Relative_luminance_and_%22gamma_encoded%22_colorspaces
  return (
    ((0.2126 * colorTrans.R + 0.7152 * colorTrans.G + 0.0722 * colorTrans.B) /
      255.0) *
    100.0
  );
}

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
        setShareUrl: (share_url) =>
          set((state) => {
            state.shareUrl = share_url;
          }),
        addGui: (guiConfig) =>
          set((state) => {
            state.guiOrderFromId[guiConfig.id] = guiConfig.props.order;
            state.guiConfigFromId[guiConfig.id] = guiConfig;
            if (!(guiConfig.container_id in state.guiIdSetFromContainerId)) {
              state.guiIdSetFromContainerId[guiConfig.container_id] = {};
            }
            state.guiIdSetFromContainerId[guiConfig.container_id]![
              guiConfig.id
            ] = true;
          }),
        addModal: (modalConfig) =>
          set((state) => {
            state.modals.push(modalConfig);
          }),
        removeModal: (id) =>
          set((state) => {
            state.modals = state.modals.filter((m) => m.id !== id);
          }),
        removeGui: (id) =>
          set((state) => {
            const guiConfig = state.guiConfigFromId[id];

            delete state.guiIdSetFromContainerId[guiConfig.container_id]![id];
            delete state.guiOrderFromId[id];
            delete state.guiConfigFromId[id];
            if (
              Object.keys(
                state.guiIdSetFromContainerId[guiConfig.container_id]!,
              ).length == 0
            )
              delete state.guiIdSetFromContainerId[guiConfig.container_id];
          }),
        resetGui: () =>
          set((state) => {
            // No need to overwrite the theme or label. The former especially
            // can be jarring.
            // state.theme = cleanGuiState.theme;
            // state.label = cleanGuiState.label;

            // This feels brittle, could be cleaned up...
            state.shareUrl = null;
            state.guiIdSetFromContainerId = {};
            state.modals = [];
            state.guiOrderFromId = {};
            state.guiConfigFromId = {};
            state.uploadsInProgress = {};
          }),
        updateUploadState: (state) =>
          set((globalState) => {
            const { componentId, ...rest } = state;
            globalState.uploadsInProgress[componentId] = {
              ...globalState.uploadsInProgress[componentId],
              ...rest,
            };
          }),
        updateGuiProps: (id, updates) => {
          set((state) => {
            const config = state.guiConfigFromId[id];
            if (config === undefined) {
              console.error("Tried to update non-existent component", id);
              return;
            }

            // Iterate over key/value pairs.
            for (const [key, value] of Object.entries(updates)) {
              // We don't put `value` in the props object to make types
              // stronger in the user-facing Python API. This results in some
              // nastiness here, we should revisit...
              if (key === "value") {
                (state.guiConfigFromId[id] as any).value = value;
              } else if (!(key in config.props)) {
                console.error(
                  `Tried to update nonexistent property '${key}' of GUI element ${id}!`,
                );
              } else {
                (state.guiConfigFromId[id].props as any)[key] = value;
              }
            }
          });
        },
      })),
    ),
  )[0];
}

/** Type corresponding to a zustand-style useGuiState hook. */
export type UseGui = ReturnType<typeof useGuiState>;
