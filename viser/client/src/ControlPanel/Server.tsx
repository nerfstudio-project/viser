import { button, LevaPanel, useControls, useCreateStore } from "leva";
import Box from "@mui/material/Box";
import { levaTheme } from "./Generated";
import { useContext, useState } from "react";
import { ViewerContext } from "..";
import { isTexture } from "../WebsocketInterface";
import { Stats } from "@react-three/drei";

export default function ServerControls() {
  const { useGui, wrapperRef, sceneRef } = useContext(ViewerContext)!;

  const [showStats, setShowStats] = useState(false);

  const server = useGui((state) => state.server);
  const label = useGui((state) => state.label);
  const backgroundAvailable = useGui((state) => state.backgroundAvailable);

  // Hack around leva bug: https://github.com/pmndrs/leva/issues/253
  // (in case the user also makes an input called Statistics!)
  const idPrefix = "_viser-server-";
  const levaStore = useCreateStore();
  useControls(
    {
      Label: {
        value: label,
        onChange: (value) => useGui.setState({ label: value }),
      },
      Websocket: {
        value: server,
        onChange: (value) => useGui.setState({ server: value }),
      },
      "Download Background": button(
        () => {
          const wrapper = wrapperRef.current;
          if (wrapper === null) return;

          if (!isTexture(sceneRef!.current!.background)) {
            // This should never happen.
            alert("No background to download!");
            return;
          }

          const data = sceneRef!.current!.background.image.src;
          console.log(data);
          const link = document.createElement("a");
          link.download = "background";
          link.href = data;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        },
        { disabled: !backgroundAvailable }
      ),
      // Note: statistics are currently global, not per-pane.
      [idPrefix + "Statistics"]: {
        label: "Statistics",
        value: showStats,
        onChange: (value) => setShowStats(value),
      },
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
      {showStats ? <Stats parent={wrapperRef} className="stats-panel" /> : null}
    </Box>
  );
}
