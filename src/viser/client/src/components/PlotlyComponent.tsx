import { GuiAddPlotlyMessage } from "../WebsocketMessages";
import { useDisclosure } from '@mantine/hooks';
import { Modal, Box, Paper, Tooltip } from '@mantine/core';

import { useEffect, useState } from "react";
import { useElementSize } from '@mantine/hooks';

// When drawing border around the plot, it should be aligned with the folder's.
import { folderWrapper } from "./Folder.css";

function generatePlotWithAspect(json_str: string, aspect_ratio: number, staticPlot: boolean) {
  // Parse json string, to construct plotly object.
  // Note that only the JSON string is kept as state, not the json object.
  const plot_json = JSON.parse(json_str);

  // This keeps the zoom-in state, etc, see https://plotly.com/javascript/uirevision/.
  plot_json.layout.uirevision = "true";

  // Box size change -> width value change -> plot rerender trigger.
  // This doesn't actually work for the *first* time a modal is opened...
  const { ref, width, height } = useElementSize();
  // Figure out if (w, h*ar) or (w/ar, h) is smaller, and choose the smaller one, to avoid overflowing.
  plot_json.layout.width = Math.min(width, height / aspect_ratio);
  plot_json.layout.height = Math.min(height, width * aspect_ratio);

  // Make the plot static, if specified.
  if (plot_json.config === undefined) plot_json.config = {};
  plot_json.config.staticPlot = staticPlot;

  // Use React hooks to update the plotly object, when the plot data changes.
  // based on https://github.com/plotly/react-plotly.js/issues/242.
  const [plotRef, setPlotRef] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    if (plotRef === null) return;
    // @ts-ignore - Plotly.js is dynamically imported with an eval() call.
    Plotly.react(
      plotRef,
      plot_json.data,
      plot_json.layout,
      plot_json.config
    );
  }, [plot_json])
  const plot_div = <div ref={setPlotRef} />

  return (
    <Paper ref={ref} className={folderWrapper} withBorder>
      {plot_div}
    </Paper>
  );
}

export default function PlotlyComponent({
  visible,
  plotly_json_str,
  aspect_ratio,
}: GuiAddPlotlyMessage) {
  if (!visible) return <></>;

  // Create two plots; one for the control panel, and one for the modal.
  // They should have different sizes, so we need to generate them separately.
  const plot_controlpanel = generatePlotWithAspect(
    plotly_json_str,
    aspect_ratio,
    true
  );
  const plot_modal = generatePlotWithAspect(
    plotly_json_str,
    aspect_ratio,
    false  // User can interact with plot in modal.
  );

  // Create a modal with the plot, and a button to open it.
  const [opened, { open, close }] = useDisclosure(false);
  return (
    <Box>
      {/* Draw static plot in the controlpanel, which can be clicked. */}
      <Tooltip.Floating
        zIndex={100}
        label={"Click to expand"}
      >
        <Box
          style={{
            cursor: "pointer",
            flexShrink: 0, position: "relative",
          }}
          onClick={open}
        >
          {plot_controlpanel}
        </Box>
      </Tooltip.Floating>

      {/* Modal contents. */}
      <Modal opened={opened} onClose={close} fullScreen>
        {plot_modal}
      </Modal>
    </Box>
  )
}