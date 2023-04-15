import { button, LevaPanel, useControls, useCreateStore } from "leva";
import Box from "@mui/material/Box";
import { levaTheme } from "./Generated";
import { UseGui } from "./GuiState";
import { RefObject } from "react";

export default function ServerControls(props: {
  useGui: UseGui;
  wrapperRef: RefObject<HTMLDivElement>;
}) {
  const server = props.useGui((state) => state.server);
  const label = props.useGui((state) => state.label);
  const backgroundAvailable = props.useGui(
    (state) => state.backgroundAvailable
  );

  const levaStore = useCreateStore();
  useControls(
    {
      Label: {
        value: label,
        onChange: (value) => props.useGui.setState({ label: value }),
      },
      Websocket: {
        value: server,
        onChange: (value) => props.useGui.setState({ server: value }),
      },
      "Download Background": button(
        () => {
          const wrapper = props.wrapperRef.current;
          if (wrapper === null) return;

          if (!wrapper.style.backgroundImage.startsWith("url(")) {
            // This should never happen.
            alert("No background to download!");
            return;
          }
          const data = wrapper.style.backgroundImage.split('"')[1];
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
