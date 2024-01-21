import React from "react";

import { FolderProps } from '../WebsocketMessages';
import { ViewerContext, ViewerContextContents } from "../App";
import {
  Collapse,
  Paper,
  Box,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { GuiGenerateContext } from "../ControlPanel/GuiState";


export default function FolderComponent(conf: FolderProps) {
  const viewer = React.useContext(ViewerContext)!;
  const { renderContainer } = React.useContext(GuiGenerateContext)!;
  const [opened, { toggle }] = useDisclosure(conf.expand_by_default);
  const guiIdSet = viewer.useGui(
    (state) => state.guiIdSetFromContainerId[conf.id],
  );
  const isEmpty = guiIdSet === undefined || Object.keys(guiIdSet).length === 0;

  const ToggleIcon = opened ? IconChevronUp : IconChevronDown;
  return (<Box pb={!last ? "0.125em" : 0}>
    <Paper
      withBorder
      pt="0.0625em"
      mx="xs"
      mt="xs"
      mb="sm"
      sx={{ position: "relative" }}
    >
      <Paper
        sx={{
          fontSize: "0.875em",
          position: "absolute",
          padding: "0 0.375em 0 0.375em",
          top: 0,
          left: "0.375em",
          transform: "translateY(-50%)",
          cursor: isEmpty ? undefined : "pointer",
          userSelect: "none",
          fontWeight: 500,
        }}
        onClick={toggle}
      >
        {conf.label}
        <ToggleIcon
          style={{
            width: "0.9em",
            height: "0.9em",
            strokeWidth: 3,
            top: "0.1em",
            position: "relative",
            marginLeft: "0.25em",
            marginRight: "-0.1em",
            opacity: 0.5,
            display: isEmpty ? "none" : undefined,
          }}
        />
      </Paper>
      <Collapse in={opened && !isEmpty} pt="0.2em">
        {renderContainer(conf.id, true)}
      </Collapse>
      <Collapse in={!(opened && !isEmpty)}>
        <Box p="xs"></Box>
      </Collapse>
    </Paper>
  </Box>
  );
}