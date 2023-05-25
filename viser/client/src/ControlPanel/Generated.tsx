import {
  button,
  buttonGroup,
  folder,
  LevaPanel,
  useControls,
  useCreateStore,
} from "leva";
import { LevaCustomTheme } from "leva/dist/declarations/src/styles";
import { UseGui } from "./GuiState";
import React, { MutableRefObject } from "react";
import Box from "@mui/material/Box";
import {
  makeThrottledMessageSender,
  sendWebsocketMessage,
} from "../WebsocketInterface";
import { ViewerContext } from "..";

export const levaTheme: LevaCustomTheme = {
  colors: {
    elevation1: "#e5e5e5",
    elevation2: "#ffffff",
    elevation3: "#f5f5f5",
    accent1: "#0066dc",
    accent2: "#1976d2",
    accent3: "#3c93ff",
    folderWidgetColor: "#777",
    highlight1: "#000000",
    highlight2: "#1d1d1d",
    highlight3: "#000000",
    vivid1: "#ffcc00",
  },
  radii: {
    xs: "2px",
    sm: "3px",
    lg: "10px",
  },
  space: {
    sm: "6px",
    md: "12px",
    rowGap: "8px",
    colGap: "8px",
  },
  fontSizes: {
    root: "0.9em",
  },
  fonts: {
    mono: "",
    sans: "",
  },
  sizes: {
    rootWidth: "350px",
    controlWidth: "170px",
    scrubberWidth: "10px",
    scrubberHeight: "14px",
    rowHeight: "24px",
    numberInputMinWidth: "60px",
    folderTitleHeight: "24px",
    checkboxSize: "16px",
    joystickWidth: "100px",
    joystickHeight: "100px",
    colorPickerWidth: "160px",
    colorPickerHeight: "100px",
    monitorHeight: "60px",
    titleBarHeight: "39px",
  },
  borderWidths: {
    root: "0px",
    input: "1px",
    focus: "1px",
    hover: "1px",
    active: "1px",
    folder: "1px",
  },
  fontWeights: {
    label: "normal",
    folder: "normal",
    button: "normal",
  },
};

/** One tab in the control panel. */
export default function GeneratedControls(props: {
  useGui: UseGui;
  websocketRef: MutableRefObject<WebSocket | null>;
}) {
  const guiNames = props.useGui((state) => state.guiNames);
  const guiConfigFromName = props.useGui((state) => state.guiConfigFromName);
  const panelKey = React.useContext(ViewerContext)!.panelKey.toString();

  // Add callbacks to guiConfigFromName.
  const suppressOnChange = React.useRef<{ [key: string]: boolean }>({});

  // We're going to try and build an object that looks like:
  // {"folder name": {"input name": leva config}}
  const guiConfigTree: { [key: string]: any } = {};

  function getFolderContainer(folderLabels: string[]) {
    let guiConfigNode = guiConfigTree;
    folderLabels.forEach((label) => {
      if (guiConfigNode[label] === undefined) {
        guiConfigNode[label] = { _is_folder_marker: true };
      }
      guiConfigNode = guiConfigNode[label];
    });
    return guiConfigNode;
  }

  guiNames.forEach((guiName) => {
    const { levaConf, folderLabels, visible } = guiConfigFromName[guiName];
    const leafFolder = getFolderContainer(folderLabels);

    // Hacky stuff that lives outside of TypeScript...
    if (levaConf["type"] === "BUTTON") {
      // Add a button.
      if (!visible) return;
      leafFolder[guiName] = button(() => {
        sendWebsocketMessage(props.websocketRef, {
          type: "GuiUpdateMessage",
          name: guiName,
          value: true,
        });
      }, levaConf["settings"]);
    } else if (levaConf["type"] === "BUTTON_GROUP") {
      // Add a button.
      if (!visible) return;
      const opts: { [key: string]: () => void } = {};
      levaConf["opts"].forEach((option: string) => {
        opts[option] = () => {
          sendWebsocketMessage(props.websocketRef, {
            type: "GuiUpdateMessage",
            name: guiName,
            value: option,
          });
        };
      });
      leafFolder[guiName] = buttonGroup({
        label: levaConf["label"],
        opts: opts,
      });
    } else {
      // Add any other kind of input.
      const sendUpdate = makeThrottledMessageSender(props.websocketRef, 50);

      // Leva uses the name of an input as its DOM id. We add the panel key as
      // a suffix to make sure every input has a unique ID; this prevents
      // interference from multiple panels connected to the same server.
      // Matters especially for checkboxes.
      //
      // This isn't applied to buttons above because buttons in Leva have no `label` field.
      leafFolder[guiName + "-" + panelKey] = {
        ...levaConf,
        onChange: (value: any, _propName: any, options: any) => {
          if (options.initial) return;
          if (suppressOnChange.current[guiName]) {
            delete suppressOnChange.current[guiName];
            return;
          }
          sendUpdate({
            type: "GuiUpdateMessage",
            name: guiName,
            value: value,
          });
        },
        render: () => visible,
      };
    }
  });

  // Recursively wrap folders in a GUI config tree with Leva's `folder()`.
  function wrapFoldersInGuiConfigTree(
    guiConfigNode: { [key: string]: any },
    root: boolean
  ) {
    const { _is_folder_marker, ...rest } = guiConfigNode;
    guiConfigNode = rest;

    if (root || _is_folder_marker === true) {
      const out: { [title: string]: any } = {};
      for (const [k, v] of Object.entries(guiConfigNode)) {
        out[k] = wrapFoldersInGuiConfigTree(v, false);
      }
      return root ? out : folder(out);
    }
    return guiConfigNode;
  }

  // Make Leva controls.
  const levaStore = useCreateStore();
  const [, set] = useControls(
    () => wrapFoldersInGuiConfigTree(guiConfigTree, true) as any,
    { store: levaStore },
    [guiConfigTree]
  );

  // Logic for setting control inputs when items are put onto the guiSetQueue.
  const guiSetQueue = props.useGui((state) => state.guiSetQueue);
  const applyGuiSetQueue = props.useGui((state) => state.applyGuiSetQueue);
  const timeouts = React.useRef<{ [key: string]: NodeJS.Timeout }>({});
  React.useEffect(() => {
    if (Object.keys(guiSetQueue).length === 0) return;
    applyGuiSetQueue((name, value) => {
      suppressOnChange.current[name] = true;

      // Suppression timeout. Resolves some issues with onChange() not firing
      // after we call set... this is hacky and should be revisited.
      clearTimeout(timeouts.current[name]);
      timeouts.current[name] = setTimeout(() => {
        suppressOnChange.current[name] = false;
      }, 10);

      // Set Leva control.
      set({ [name + "-" + panelKey]: value });
    });
  }, [guiSetQueue, applyGuiSetQueue, set]);

  // Leva theming is a bit limited, so we hack at styles here...
  return (
    <Box
      component="div"
      sx={{
        "& label": { color: "#777" },
        "& input[type='checkbox']~label svg path": {
          stroke: "#fff !important",
        },
        "& button:not(:only-child)": {
          // Button groups.
          color: "#777 !important",
          backgroundColor: "#e5e5e5 !important",
        },
        "& button:only-child": {
          // Single buttons.
          color: "#fff !important",
          height: "2em",
        },
      }}
    >
      <LevaPanel
        fill
        flat
        titleBar={false}
        theme={levaTheme}
        store={levaStore}
        hideCopyButton
      />
    </Box>
  );
}
