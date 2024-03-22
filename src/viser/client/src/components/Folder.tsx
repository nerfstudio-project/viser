import * as React from "react";
import { useDisclosure } from "@mantine/hooks";
import { GuiAddFolderMessage } from "../WebsocketMessages";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { Box, Collapse, Paper } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViewerContext } from "../App";

export default function FolderComponent({
  id,
  label,
  visible,
  expand_by_default,
}: GuiAddFolderMessage) {
  const viewer = React.useContext(ViewerContext)!;
  const [opened, { toggle }] = useDisclosure(expand_by_default);
  const guiIdSet = viewer.useGui((state) => state.guiIdSetFromContainerId[id]);
  const guiContext = React.useContext(GuiComponentContext)!;
  const isEmpty = guiIdSet === undefined || Object.keys(guiIdSet).length === 0;

  const ToggleIcon = opened ? IconChevronUp : IconChevronDown;
  if (!visible) return <></>;
  return (
    <Paper
      withBorder
      pt="0.0625em"
      mb="xs"
      mx="xs"
      mt="xs"
      sx={{
        position: "relative",
        ":not(:last-child)": { marginBottom: "1em" },
      }}
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
        {label}
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
        <GuiComponentContext.Provider
          value={{
            ...guiContext,
            folderDepth: guiContext.folderDepth + 1,
          }}
        >
          <guiContext.GuiContainer containerId={id} />
        </GuiComponentContext.Provider>
      </Collapse>
      <Collapse in={!(opened && !isEmpty)}>
        <Box p="xs"></Box>
      </Collapse>
    </Paper>
  );
}
