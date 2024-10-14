import * as React from "react";
import { useDisclosure } from "@mantine/hooks";
import { GuiFolderMessage } from "../WebsocketMessages";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { Box, Collapse, Paper } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViewerContext } from "../App";
import { folderLabel, folderToggleIcon, folderWrapper } from "./Folder.css";

export default function FolderComponent({
  uuid,
  props: { label, visible, expand_by_default },
}: GuiFolderMessage) {
  const viewer = React.useContext(ViewerContext)!;
  const [opened, { toggle }] = useDisclosure(expand_by_default);
  const guiIdSet = viewer.useGui(
    (state) => state.guiUuidSetFromContainerUuid[uuid],
  );
  const guiContext = React.useContext(GuiComponentContext)!;
  const isEmpty = guiIdSet === undefined || Object.keys(guiIdSet).length === 0;

  const ToggleIcon = opened ? IconChevronUp : IconChevronDown;
  if (!visible) return <></>;
  return (
    <Paper withBorder className={folderWrapper}>
      <Paper
        className={folderLabel}
        style={{
          cursor: isEmpty ? undefined : "pointer",
        }}
        onClick={toggle}
      >
        {label}
        <ToggleIcon
          className={folderToggleIcon}
          style={{
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
          <guiContext.GuiContainer containerUuid={uuid} />
        </GuiComponentContext.Provider>
      </Collapse>
      <Collapse in={!(opened && !isEmpty)}>
        <Box p="xs"></Box>
      </Collapse>
    </Paper>
  );
}
