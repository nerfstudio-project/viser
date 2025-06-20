import { useEffect, useRef, useMemo } from "react";
import uPlot from "uplot";
import UplotReact from "uplot-react";
import "uplot/dist/uPlot.min.css";

import { Modal, Box, Paper, Tooltip } from "@mantine/core";
import { useDisclosure, useElementSize } from "@mantine/hooks";
import { GuiUplotMessage } from "../WebsocketMessages";
import { folderWrapper } from "./Folder.css";

export default function UplotComponent({
  props: {
    data,
    series,
    title,
    mode,
    bands,
    scales,
    axes,
    legend,
    cursor,
    focus,
    aspect,
  },
}: GuiUplotMessage) {
  // Modal state
  const [opened, { open, close }] = useDisclosure(false);

  // Container sizing
  const { ref: containerSizeRef, width: containerWidth } = useElementSize();
  const { ref: modalContainerSizeRef, width: modalContainerWidth } =
    useElementSize();

  // uPlot instance refs for both small and modal plots
  const plotRef = useRef<uPlot | null>(null);
  const modalPlotRef = useRef<uPlot | null>(null);
  const lastCursorPos = useRef<{ left: number; top: number } | null>(null);

  // Convert inputs to Float64Array once per update
  const alignedData = useMemo(() => {
    const convertedData = data.map((array: Uint8Array) => {
      return new Float64Array(
        array.buffer.slice(
          array.byteOffset,
          array.byteOffset + array.byteLength,
        ),
      );
    });
    return convertedData;
  }, [data]);

  // Build base uPlot options from the props
  const baseUplotOptions = useMemo(() => {
    const options: any = {
      title: title || undefined,
      mode: mode || undefined,
      series: series || [],
    };
    if (bands !== null) options.bands = bands;
    if (scales !== null) options.scales = scales;
    if (axes !== null) options.axes = axes;
    if (legend !== null) options.legend = legend;
    if (cursor !== null) options.cursor = cursor;
    if (focus !== null) options.focus = focus;

    return options;
  }, [title, mode, series, bands, scales, axes, legend, cursor, focus]);

  // Small plot options (for the preview)
  const smallPlotOptions = useMemo(() => {
    if (containerWidth <= 0) return undefined;
    return {
      width: containerWidth,
      height: containerWidth * aspect,
      cursor: {
        show: true,
        drag: { setScale: true },
        points: { show: true, size: 4 },
        ...baseUplotOptions.cursor,
      },
      ...baseUplotOptions,
    };
  }, [containerWidth, aspect, baseUplotOptions]);

  // Modal plot options (for the expanded view)
  const modalPlotOptions = useMemo(() => {
    if (modalContainerWidth <= 0) return undefined;
    return {
      width: modalContainerWidth,
      height: modalContainerWidth * aspect,
      cursor: {
        show: true,
        drag: { setScale: true },
        points: { show: true, size: 4 },
        ...baseUplotOptions.cursor,
      },
      ...baseUplotOptions,
    };
  }, [modalContainerWidth, aspect, baseUplotOptions]);

  // Handle plot lifecycle for small plot
  const handleSmallPlotCreate = (chart: uPlot) => {
    plotRef.current = chart;
    if (lastCursorPos.current) {
      chart.setCursor(lastCursorPos.current);
    }
  };

  const handleSmallPlotDelete = (chart: uPlot) => {
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

  // Handle plot lifecycle for modal plot
  const handleModalPlotCreate = (chart: uPlot) => {
    modalPlotRef.current = chart;
    if (lastCursorPos.current) {
      chart.setCursor(lastCursorPos.current);
    }
  };

  const handleModalPlotDelete = (chart: uPlot) => {
    if (modalPlotRef.current === chart) {
      if (chart.cursor.left != null && chart.cursor.top != null) {
        lastCursorPos.current = {
          left: chart.cursor.left,
          top: chart.cursor.top,
        };
      }
      modalPlotRef.current = null;
    }
  };

  // Update data for both plots
  useEffect(() => {
    if (plotRef.current) {
      plotRef.current.setData(alignedData);
    }
    if (opened && modalPlotRef.current) {
      modalPlotRef.current.setData(alignedData);
    }
  }, [alignedData, opened]);

  return (
    <Box>
      <Tooltip.Floating label="Click to expand" zIndex={100}>
        <Box onClick={open} style={{ cursor: "pointer", flexShrink: 0 }}>
          <Paper
            ref={containerSizeRef}
            className={folderWrapper}
            withBorder
            style={{ position: "relative" }}
          >
            {smallPlotOptions && (
              <UplotReact
                options={smallPlotOptions}
                data={alignedData}
                onCreate={handleSmallPlotCreate}
                onDelete={handleSmallPlotDelete}
              />
            )}
          </Paper>
        </Box>
      </Tooltip.Floating>

      <Modal opened={opened} onClose={close} size="xl" keepMounted>
        <Paper
          ref={modalContainerSizeRef}
          className={folderWrapper}
          withBorder
          style={{ position: "relative" }}
        >
          {modalPlotOptions && (
            <UplotReact
              options={modalPlotOptions}
              data={alignedData}
              onCreate={handleModalPlotCreate}
              onDelete={handleModalPlotDelete}
            />
          )}
        </Paper>
      </Modal>
    </Box>
  );
}
