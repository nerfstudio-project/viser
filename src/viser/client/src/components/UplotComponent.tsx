import { useMemo, useState } from "react";
import UplotReact from "uplot-react";
import "uplot/dist/uPlot.min.css";

import { Modal, Box, Paper, Tooltip, ActionIcon } from "@mantine/core";
import { useDisclosure, useElementSize } from "@mantine/hooks";
import { IconMaximize } from "@tabler/icons-react";
import { GuiUplotMessage } from "../WebsocketMessages";
import { folderWrapper } from "./Folder.css";

// Individual plot component.
function PlotComponent({
  props,
  onExpand,
}: GuiUplotMessage & {
  onExpand?: () => void;
}) {
  const [isHovered, setIsHovered] = useState(false);
  const { ref: containerSizeRef, width: containerWidth } = useElementSize();

  // Convert inputs to Float64Array once per update.
  const data = useMemo(() => {
    const convertedData = props.data.map((array: Uint8Array) => {
      return new Float64Array(
        array.buffer.slice(
          array.byteOffset,
          array.byteOffset + array.byteLength,
        ),
      );
    });
    return convertedData;
  }, [props.data]);

  // Build uPlot options from the props.
  //
  // There are some `any` casts because the types here come through multiple
  // transpiler layers, which are imperfect: TS=>Python=>TS.
  const plotOptions = useMemo(() => {
    return {
      width: containerWidth,
      height: (containerWidth / props.aspect) as any,
      title: props.title || undefined,
      mode: props.mode || undefined,
      series: (props.series as any) || [],
      cursor: (props.cursor as any) || undefined,
      bands: props.bands || undefined,
      scales: props.scales || undefined,
      axes: (props.axes as any) || undefined,
      legend: (props.legend as any) || undefined,
      focus: props.focus || undefined,
    };
  }, [containerWidth, props]);

  return (
    <Paper
      ref={containerSizeRef}
      className={folderWrapper}
      withBorder
      style={{ position: "relative" }}
      onMouseEnter={onExpand ? () => setIsHovered(true) : undefined}
      onMouseLeave={onExpand ? () => setIsHovered(false) : undefined}
    >
      {plotOptions && <UplotReact options={plotOptions} data={data} />}
      {onExpand && isHovered && (
        <Tooltip label="Expand plot">
          <ActionIcon
            onClick={onExpand}
            variant="subtle"
            color="gray"
            size="sm"
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              backgroundColor: "rgba(255, 255, 255, 0.9)",
              backdropFilter: "blur(4px)",
            }}
          >
            <IconMaximize size={14} />
          </ActionIcon>
        </Tooltip>
      )}
    </Paper>
  );
}

export default function UplotComponent(message: GuiUplotMessage) {
  // Modal state.
  const [opened, { open, close }] = useDisclosure(false);

  // Visibility check
  if (message.props.visible === false) return null;

  return (
    <Box>
      {/* Small plot with expand button. */}
      <PlotComponent {...message} onExpand={open} />

      {/* Modal with larger plot. */}
      <Modal opened={opened} onClose={close} size="xl">
        <PlotComponent {...message} />
      </Modal>
    </Box>
  );
}
