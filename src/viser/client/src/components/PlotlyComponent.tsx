import React from "react";
import { GuiPlotlyMessage } from "../WebsocketMessages";
import { useDisclosure } from "@mantine/hooks";
import { Modal, Box, Paper, Tooltip, ActionIcon } from "@mantine/core";
import { useElementSize } from "@mantine/hooks";
import { IconMaximize } from "@tabler/icons-react";

// When drawing border around the plot, it should be aligned with the folder's.
import { folderWrapper } from "./Folder.css";

const PlotWithAspect = React.memo(function PlotWithAspect({
  jsonStr,
  aspectRatio,
  onExpand,
}: {
  jsonStr: string;
  aspectRatio: number;
  onExpand?: () => void;
}) {
  // Hover state for expand button
  const [isHovered, setIsHovered] = React.useState(false);

  // Catch if the jsonStr is empty; if so, render an empty div.
  if (jsonStr === "") return <div></div>;

  // Parse json string, to construct plotly object.
  // Note that only the JSON string is kept as state, not the json object.
  const plotJson = JSON.parse(jsonStr);

  // This keeps the zoom-in state, etc, see https://plotly.com/javascript/uirevision/.
  plotJson.layout.uirevision = "true";

  // Box size change -> width value change -> plot rerender trigger.
  const { ref, width } = useElementSize();
  plotJson.layout.width = width;
  plotJson.layout.height = width * aspectRatio;

  // Use React hooks to update the plotly object, when the plot data changes.
  // based on https://github.com/plotly/react-plotly.js/issues/242.
  const plotRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    // @ts-ignore - Plotly.js is dynamically imported with an eval() call.
    Plotly.react(
      plotRef.current!,
      plotJson.data,
      plotJson.layout,
      plotJson.config,
    );
  }, [plotJson]);

  return (
    <Paper
      ref={ref}
      className={folderWrapper}
      withBorder
      style={{ position: "relative" }}
      onMouseEnter={onExpand ? () => setIsHovered(true) : undefined}
      onMouseLeave={onExpand ? () => setIsHovered(false) : undefined}
    >
      <div ref={plotRef} />
      {/* Show expand icon on hover */}
      {onExpand && isHovered && (
        <Tooltip label="Expand plot">
          <ActionIcon
            onClick={onExpand}
            variant="subtle"
            color="gray"
            size="sm"
            style={{
              position: "absolute",
              bottom: 8,
              right: 8,
              backgroundColor: "rgba(255, 255, 255, 0.9)",
              backdropFilter: "blur(4px)",
              zIndex: 1001,
            }}
          >
            <IconMaximize size={14} />
          </ActionIcon>
        </Tooltip>
      )}
    </Paper>
  );
});

export default function PlotlyComponent({
  props: { visible, _plotly_json_str: plotly_json_str, aspect },
}: GuiPlotlyMessage) {
  if (!visible) return null;

  // Create a modal with the plot, and a button to open it.
  const [opened, { open, close }] = useDisclosure(false);
  return (
    <Box>
      {/* Draw interactive plot in the controlpanel with hover-to-expand icon */}
      <PlotWithAspect
        jsonStr={plotly_json_str}
        aspectRatio={aspect}
        onExpand={open}
      />

      {/* Modal contents. */}
      <Modal opened={opened} onClose={close} size="xl">
        <PlotWithAspect jsonStr={plotly_json_str} aspectRatio={aspect} />
      </Modal>
    </Box>
  );
}
