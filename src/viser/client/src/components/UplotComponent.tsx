import React, { useEffect, useRef, useMemo } from "react";
import uPlot from "uplot";
import UplotReact from "uplot-react";
import "uplot/dist/uPlot.min.css";

import { Modal, Box, Paper, Tooltip } from "@mantine/core";
import { useDisclosure, useElementSize } from "@mantine/hooks";
import { GuiUplotMessage } from "../WebsocketMessages";
import { folderWrapper } from "./Folder.css";


const PlotData = React.memo(function PlotData({
  data,
  options,
  aspectRatio = 1.0,
  isVisible = true,
}: {
  data: number[][]; // managed by React
  options: { [key: string]: any };
  aspectRatio?: number;
  isVisible?: boolean;
}) {
  const { ref: containerSizeRef, width: containerWidth } = useElementSize();
  const plotRef = useRef<uPlot | null>(null);
  const lastCursorPos = useRef<{ left: number; top: number } | null>(null);

  // Save cursor before destroying plot
  const handleDelete = (chart: uPlot) => {
    if (plotRef.current === chart) {
      if (chart.cursor.left != null && chart.cursor.top != null) {
        lastCursorPos.current = {
          left: chart.cursor.left,
          top: chart.cursor.top,
        };
      }
      plotRef.current = null;
    }
  };

  // Restore cursor after creating plot
  const handleCreate = (chart: uPlot) => {
    plotRef.current = chart;
    if (lastCursorPos.current) {
      chart.setCursor(lastCursorPos.current);
    }
  };

  // Get fresh options when container size changes
  const uplotOptions = useMemo(() => {
    if (containerWidth <= 0) return undefined;
    return {
      width: containerWidth,
      height: containerWidth * aspectRatio,
      cursor: {
        show: true,
        drag: { setScale: true },
        points: { show: true, size: 4 },
      },
      ...options,
    };
  }, [containerWidth, aspectRatio, options]);

  // Update data (does not reset cursor)
  useEffect(() => {
    if (!isVisible || !plotRef.current) return;
    plotRef.current.setData(data);
  }, [data, isVisible]);

  return (
    <Paper
      ref={containerSizeRef}
      className={folderWrapper}
      withBorder
      style={{ position: "relative" }}
    >
      {uplotOptions && (
        <UplotReact
          options={uplotOptions}
          data={data}
          onCreate={handleCreate}
          onDelete={handleDelete}
        />
      )}
    </Paper>
  );
});


export default function UplotComponent({
  props: { aligned_data, options, aspect },
}: GuiUplotMessage) {

  // Create a modal with the plot, and a button to open it.
  const [opened, { open, close }] = useDisclosure(false);

  // Convert inputs to Float32Array once per update
  const alignedData = useMemo<uPlot.AlignedData>(() => {
    const traj = aligned_data.map((traj) => new Float32Array(traj));
    return [...traj];
  }, [aligned_data]);

  return (
    <Box>
      <Tooltip.Floating label="Click to expand" zIndex={100}>
        <Box onClick={open} style={{ cursor: "pointer", flexShrink: 0 }}>
          <PlotData
            data={alignedData}
            options={options}
            aspectRatio={aspect}
            isVisible={true}
          />
        </Box>
      </Tooltip.Floating>

      <Modal opened={opened} onClose={close} size="xl" keepMounted>
        <PlotData
          data={alignedData}
          options={options}
          aspectRatio={aspect}
          isVisible={opened}
        />
      </Modal>
    </Box>
  );
}