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
  showOrbitOriginTool: boolean;
  guiUuidSetFromContainerUuid: {
    [containerUuid: string]: { [uuid: string]: true } | undefined;
  };
  modals: GuiModalMessage[];
  guiOrderFromUuid: { [id: string]: number };
  guiConfigFromUuid: { [id: string]: GuiComponentMessage | undefined };
  uploadsInProgress: {
    [uuid: string]: {
      notificationId: string;
      uploadedBytes: number;
      totalBytes: number;
      filename: string;
    };
  };
  cameraEnabled: boolean;
  cameraReady: boolean;
  activeCameraRequest: {
    request_id: string;
    max_resolution: number | null;
    facing_mode: "user" | "environment" | null;
  } | null;
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
  setCameraEnabled: (enabled: boolean) => void;
  setCameraReady: (ready: boolean) => void;
  setCameraRequest: (
    request: {
      request_id: string;
      max_resolution: number | null;
      facing_mode: "user" | "environment" | null;
    } | null
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
  showOrbitOriginTool: false,
  guiUuidSetFromContainerUuid: {},
  modals: [],
  guiOrderFromUuid: {},
  guiConfigFromUuid: {},
  uploadsInProgress: {},
  cameraEnabled: false,
  cameraReady: false,
  activeCameraRequest: null,
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
            state.guiOrderFromUuid[guiConfig.uuid] = guiConfig.props.order;
            state.guiConfigFromUuid[guiConfig.uuid] = guiConfig;
            if (
              !(guiConfig.container_uuid in state.guiUuidSetFromContainerUuid)
            ) {
              state.guiUuidSetFromContainerUuid[guiConfig.container_uuid] = {};
            }
            state.guiUuidSetFromContainerUuid[guiConfig.container_uuid]![
              guiConfig.uuid
            ] = true;
          }),
        addModal: (modalConfig) =>
          set((state) => {
            state.modals.push(modalConfig);
          }),
        removeModal: (id) =>
          set((state) => {
            state.modals = state.modals.filter((m) => m.uuid !== id);
          }),
        removeGui: (id) =>
          set((state) => {
            const guiConfig = state.guiConfigFromUuid[id];
            if (guiConfig == undefined) {
              // TODO: this will currently happen when GUI elements are removed
              // and then a new client connects. Needs to be revisited.
              console.warn("(OK) Tried to remove non-existent component", id);
              return;
            }
            delete state.guiUuidSetFromContainerUuid[guiConfig.container_uuid]![
              id
            ];
            delete state.guiOrderFromUuid[id];
            delete state.guiConfigFromUuid[id];
            if (
              Object.keys(
                state.guiUuidSetFromContainerUuid[guiConfig.container_uuid]!,
              ).length == 0
            )
              delete state.guiUuidSetFromContainerUuid[
                guiConfig.container_uuid
              ];
          }),
        resetGui: () =>
          set((state) => {
            // No need to overwrite the theme or label. The former especially
            // can be jarring.
            // state.theme = cleanGuiState.theme;
            // state.label = cleanGuiState.label;

            // This feels brittle, could be cleaned up...
            state.shareUrl = cleanGuiState.shareUrl;
            state.guiUuidSetFromContainerUuid =
              cleanGuiState.guiUuidSetFromContainerUuid;
            state.modals = cleanGuiState.modals;
            state.guiOrderFromUuid = cleanGuiState.guiOrderFromUuid;
            state.guiConfigFromUuid = cleanGuiState.guiConfigFromUuid;
            state.uploadsInProgress = cleanGuiState.uploadsInProgress;
            state.cameraEnabled = cleanGuiState.cameraEnabled;
            state.cameraReady = cleanGuiState.cameraReady;
            state.activeCameraRequest = cleanGuiState.activeCameraRequest;
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
            const config = state.guiConfigFromUuid[id];
            if (config === undefined) {
              console.error(
                `Tried to update non-existent component '${id}' with`,
                updates,
              );
              return;
            }

            // Iterate over key/value pairs.
            for (const [key, value] of Object.entries(updates)) {
              // We don't put `value` in the props object to make types
              // stronger in the user-facing Python API. This results in some
              // nastiness here, we should revisit...
              if (key === "value") {
                (state.guiConfigFromUuid[id] as any).value = value;
              } else if (!(key in config.props)) {
                console.error(
                  `Tried to update nonexistent property '${key}' of GUI element ${id}!`,
                );
              } else {
                (config.props as any)[key] = value;
              }
            }
          });
        },
        setCameraEnabled: (enabled) =>
          set((state) => {
            state.cameraEnabled = enabled;
            if (!enabled) {
              state.cameraReady = false;
              state.activeCameraRequest = null;
            }
          }),
        setCameraReady: (ready) =>
          set((state) => {
            state.cameraReady = ready;
          }),
        setCameraRequest: (request) =>
          set((state) => {
            state.activeCameraRequest = request;
          }),
      })),
    ),
  )[0];
}

/** Type corresponding to a zustand-style useGuiState hook. */
export type UseGui = ReturnType<typeof useGuiState>;
