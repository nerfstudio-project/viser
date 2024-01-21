import * as Messages from "../WebsocketMessages";
import React from "react";
import { create } from "zustand";
import { ColorTranslator } from "colortranslator";

import { immer } from "zustand/middleware/immer";
import { ViewerContext } from "../App";
import { MantineThemeOverride } from "@mantine/core";


export type GuiGenerateContextProps<T extends Omit<Messages.AllComponentProps, "type">> = {
  update: (changes: Partial<T>) => void;
  renderContainer: (containerId: string, incrementFolderDepth?: boolean) => React.ReactNode;
  folderDepth: number;
};
export const GuiGenerateContext = React.createContext<GuiGenerateContextProps<any> | null>(null);


export type GuiProps<T extends Messages.AllComponentProps> = Omit<T, "type"> & GuiGenerateContextProps<Omit<T, "type">> & { id: string };

interface GuiState {
  theme: Messages.ThemeConfigurationMessage;
  label: string;
  server: string;
  websocketConnected: boolean;
  backgroundAvailable: boolean;
  guiIdSetFromContainerId: {
    [containerId: string]: { [id: string]: true } | undefined;
  };
  modals: Messages.GuiModalMessage[];
  guiOrderFromId: { [id: string]: number };
  guiConfigFromId: { [id: string]: Messages.AllComponentProps & { container_id: string } };
  guiCallbackFromId: { [id: string]: { [callback: string]: (value: any) => void } };
  guiValueFromId: { [id: string]: any };
  guiAttributeFromId: {
    [id: string]: { visible?: boolean; disabled?: boolean } | undefined;
  };
}

interface GuiActions {
  setTheme: (theme: Messages.ThemeConfigurationMessage) => void;
  addGui: (config: Messages.AllComponentProps & { id: string, order: number, container_id: string }) => void;
  setGuiValue: (id: string, value: any) => void;
  setGuiVisible: (id: string, visible: boolean) => void;
  setGuiDisabled: (id: string, visible: boolean) => void;
  removeGui: (id: string) => void;
  resetGui: () => void;

  dispatchCallback: (id: string, callback: string, value: any) => void;
}

const cleanGuiState: GuiState = {
  theme: {
    type: "ThemeConfigurationMessage",
    titlebar_content: null,
    control_layout: "floating",
    control_width: "medium",
    dark_mode: false,
    show_logo: true,
    colors: null,
  },
  label: "",
  server: "ws://localhost:8080", // Currently this will always be overridden.
  websocketConnected: false,
  backgroundAvailable: false,
  guiIdSetFromContainerId: {},
  modals: [],
  guiOrderFromId: {},
  guiConfigFromId: {},
  guiValueFromId: {},
  guiAttributeFromId: {},
  guiCallbackFromId: {},
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
        addGui: ({id, order, ...props }: Messages.AllComponentProps & { 
            id: string, 
            order: number, 
            container_id: string }) =>
          set((state) => {
            if (props.type == "Modal") {
              state.modals.push(modalConfig);
            } else {
              state.guiOrderFromId[id] = order;
              state.guiConfigFromId[id] = props;
              state.guiCallbackFromId[id] = {};
              if (!(props.container_id in state.guiIdSetFromContainerId)) {
                state.guiIdSetFromContainerId[props.container_id] = {};
              }
              state.guiIdSetFromContainerId[props.container_id]![id] = true;
            }
          }),
        removeModal: (id) =>
          set((state) => {
            state.modals = state.modals.filter((m) => m.id !== id);
          }),
        setGuiValue: (id, value) =>
          set((state) => {
            state.guiValueFromId[id] = value;
          }),
        setGuiVisible: (id, visible) =>
          set((state) => {
            state.guiAttributeFromId[id] = {
              ...state.guiAttributeFromId[id],
              visible: visible,
            };
          }),
        setGuiDisabled: (id, disabled) =>
          set((state) => {
            state.guiAttributeFromId[id] = {
              ...state.guiAttributeFromId[id],
              disabled: disabled,
            };
          }),
        removeGui: (id) =>
          set((state) => {
            const guiConfig = state.guiConfigFromId[id];

            delete state.guiIdSetFromContainerId[guiConfig.container_id]![id];
            delete state.guiOrderFromId[id];
            delete state.guiConfigFromId[id];
            delete state.guiValueFromId[id];
            delete state.guiAttributeFromId[id];
          }),
        resetGui: () =>
          set((state) => {
            state.guiIdSetFromContainerId = {};
            state.guiOrderFromId = {};
            state.guiConfigFromId = {};
            state.guiValueFromId = {};
            state.guiAttributeFromId = {};
          }),
        dispatchCallback: (id, callback, value) =>
          set((state) => {
            if (id in state.guiCallbackFromId && callback in state.guiCallbackFromId[id]) {
              state.guiCallbackFromId[id][callback](value)
            }
          }),
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
