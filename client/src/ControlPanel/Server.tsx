import { button, LevaPanel, useControls, useCreateStore } from "leva";
import Box from "@mui/material/Box";
import { levaTheme } from "./Generated";
import { UseGui } from "./GuiState";
import React, { RefObject } from "react";
import { Button, FormControl, InputLabel, OutlinedInput } from "@mui/material";

interface ServerControlsProps {
  useGui: UseGui;
  wrapperRef: RefObject<HTMLDivElement>;
}

export default function ServerControls(props: ServerControlsProps) {
  const setServer = props.useGui((state) => state.setServer);
  const setLabel = props.useGui((state) => state.setLabel);
  const server = props.useGui((state) => state.server);
  const label = props.useGui((state) => state.label);
  const backgroundAvailable = props.useGui(
    (state) => state.backgroundAvailable
  );

  const levaStore = useCreateStore();
  useControls(
    {
      Label: { value: label, onChange: setLabel },
      URL: { value: server, onChange: setServer },
      "Download Background": button(
        (get) => {
          if (
            !props.wrapperRef.current!.style.backgroundImage.startsWith("url(")
          ) {
            // TODO: we should consider hiding this button if there's no background available.
            alert("No background to download!");
            return;
          }
          const data =
            props.wrapperRef.current!.style.backgroundImage.split('"')[1];
          const link = document.createElement("a");
          link.download = "background";
          link.href = data;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        },
        { disabled: !backgroundAvailable }
      ),
    },
    { store: levaStore },
    [backgroundAvailable]
  );

  // Leva theming is a bit limited, so we hack at styles here...
  return (
    <Box
      component="div"
      sx={{
        "& label": { color: "#777" },
        "& input[type='checkbox']~label svg path": {
          stroke: "#fff !important",
        },
        "& button": { color: "#fff !important", height: "2em" },
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
