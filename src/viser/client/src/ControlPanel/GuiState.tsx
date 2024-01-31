import * as Messages from "../WebsocketMessages";
import React from "react";
import { create } from "zustand";
import { ColorTranslator } from "colortranslator";

import { immer } from "zustand/middleware/immer";
import { ViewerContext } from "../App";
import { MantineThemeOverride } from "@mantine/core";

export type GuiConfig = Messages.GuiAddComponentMessage;

export function isGuiConfig(message: Messages.Message): message is GuiConfig {
  return message.type.startsWith("GuiAdd");
}

interface GuiState {
  theme: Messages.ThemeConfigurationMessage;
  label: string;
  server: string;
  shareUrl: string | null;
  websocketConnected: boolean;
  backgroundAvailable: boolean;
  guiIdSetFromContainerId: {
    [containerId: string]: { [id: string]: true } | undefined;
  };
  modals: Messages.GuiModalMessage[];
  guiOrderFromId: { [id: string]: number };
  guiConfigFromId: { [id: string]: GuiConfig };
}

interface GuiActions {
  setTheme: (theme: Messages.ThemeConfigurationMessage) => void;
  setShareUrl: (share_url: string | null) => void;
  addGui: (config: GuiConfig) => void;
  addModal: (config: Messages.GuiModalMessage) => void;
  removeModal: (id: string) => void;
  updateGuiProps: (id: string, changes: Messages.GuiComponentPropsPartial) => Messages.GuiAddComponentMessage;
  removeGui: (id: string) => void;
  resetGui: () => void;
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
      immer<GuiState & GuiActions>((set, get) => ({
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
            state.guiOrderFromId[guiConfig.id] = guiConfig.order;
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
          }),
        resetGui: () =>
          set((state) => {
            state.shareUrl = null;
            state.guiIdSetFromContainerId = {};
            state.guiOrderFromId = {};
            state.guiConfigFromId = {};
          }),
        updateGuiProps: (id, changes) => {
          set((state) => {
            const config = state.guiConfigFromId[id];
            if (config === undefined) return;
            state.guiConfigFromId[id] = {...config, ...changes} as GuiConfig;
          });
          return get().guiConfigFromId[id];
        }
      })),
    ),
  )[0];
}

export function useViserMantineTheme(): MantineThemeOverride {
  const viewer = React.useContext(ViewerContext)!;
  const colors = viewer.useGui((state) => state.theme.colors);

  return {
    colorScheme: viewer.useGui((state) => state.theme.dark_mode)
      ? "dark"
      : "light",
    primaryColor: colors === null ? undefined : "custom",
    colors: {
      default: [
        "#f3f3fe",
        "#e4e6ed",
        "#c8cad3",
        "#a9adb9",
        "#9093a4",
        "#808496",
        "#767c91",
        "#656a7e",
        "#585e72",
        "#4a5167",
      ],
      ...(colors === null
        ? undefined
        : {
            custom: colors,
          }),
    },
    fontFamily: "Inter",
    components: {
      Checkbox: {
        defaultProps: {
          radius: "xs",
        },
      },
      ColorInput: {
        defaultProps: {
          radius: "xs",
        },
      },
      Select: {
        defaultProps: {
          radius: "sm",
        },
      },
      TextInput: {
        defaultProps: {
          radius: "xs",
        },
      },
      NumberInput: {
        defaultProps: {
          radius: "xs",
        },
      },
      Paper: {
        defaultProps: {
          radius: "xs",
        },
      },
      ActionIcon: {
        defaultProps: {
          radius: "xs",
        },
      },
      Button: {
        defaultProps: {
          radius: "xs",
        },
        variants: {
          filled: (theme) => ({
            root: {
              fontWeight: 450,
              color:
                computeRelativeLuminance(theme.fn.primaryColor()) > 50.0
                  ? theme.colors.gray[9] + " !important"
                  : theme.white,
            },
          }),
        },
      },
    },
  };
}

/** Type corresponding to a zustand-style useGuiState hook. */
export type UseGui = ReturnType<typeof useGuiState>;
