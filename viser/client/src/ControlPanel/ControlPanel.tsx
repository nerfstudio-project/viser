import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import GeneratedGuiContainer from "./Generated";
import { ViewerContext } from "../App";
import ServerControls from "./ServerControls";
import {
  ActionIcon,
  Box,
  Collapse,
  Tooltip,
  useMantineTheme,
} from "@mantine/core";
import {
  IconAdjustments,
  IconCloudCheck,
  IconCloudOff,
  IconArrowBack,
} from "@tabler/icons-react";
import React from "react";
import BottomPanel from "./BottomPanel";
import FloatingPanel from "./FloatingPanel";
import { ThemeConfigurationMessage } from "../WebsocketMessages";
import SidebarPanel from "./SidebarPanel";

// Must match constant in Python.
const ROOT_CONTAINER_ID = "root";

export default function ControlPanel(props: {
  control_layout: ThemeConfigurationMessage["control_layout"];
}) {
  const theme = useMantineTheme();
  const useMobileView = useMediaQuery(`(max-width: ${theme.breakpoints.xs})`);

  // TODO: will result in unnecessary re-renders.
  const viewer = React.useContext(ViewerContext)!;
  const showGenerated = viewer.useGui(
    (state) => "root" in state.guiIdSetFromContainerId,
  );
  const [showSettings, { toggle }] = useDisclosure(false);

  const generatedServerToggleButton = (
    <Box sx={{ display: showGenerated ? undefined : "none" }}>
      <ActionIcon
        onClick={(evt) => {
          evt.stopPropagation();
          toggle();
        }}
      >
        <Tooltip
          label={showSettings ? "Return to GUI" : "Connection & diagnostics"}
        >
          {showSettings ? (
            <IconArrowBack stroke={1.625} />
          ) : (
            <IconAdjustments stroke={1.625} />
          )}
        </Tooltip>
      </ActionIcon>
    </Box>
  );

  const panelContents = (
    <>
      <Collapse in={!showGenerated || showSettings} p="xs">
        <ServerControls />
      </Collapse>
      <Collapse in={showGenerated && !showSettings}>
        <GeneratedGuiContainer containerId={ROOT_CONTAINER_ID} />
      </Collapse>
    </>
  );

  if (useMobileView) {
    /* Mobile layout. */
    return (
      <BottomPanel>
        <BottomPanel.Handle>
          <ConnectionStatus />
          {generatedServerToggleButton}
        </BottomPanel.Handle>
        <BottomPanel.Contents>{panelContents}</BottomPanel.Contents>
      </BottomPanel>
    );
  } else if (props.control_layout === "floating") {
    /* Floating layout. */
    return (
      <FloatingPanel>
        <FloatingPanel.Handle>
          <ConnectionStatus />
          <FloatingPanel.HideWhenCollapsed>
            {generatedServerToggleButton}
          </FloatingPanel.HideWhenCollapsed>
        </FloatingPanel.Handle>
        <FloatingPanel.Contents>{panelContents}</FloatingPanel.Contents>
      </FloatingPanel>
    );
  } else {
    /* Sidebar view. */
    return (
      <SidebarPanel collapsible={props.control_layout === "collapsible"}>
        <SidebarPanel.Handle>
          <ConnectionStatus />
          {generatedServerToggleButton}
        </SidebarPanel.Handle>
        <SidebarPanel.Contents>{panelContents}</SidebarPanel.Contents>
      </SidebarPanel>
    );
  }
}

/* Icon and label telling us the current status of the websocket connection. */
function ConnectionStatus() {
  const { useGui } = React.useContext(ViewerContext)!;
  const connected = useGui((state) => state.websocketConnected);
  const server = useGui((state) => state.server);
  const label = useGui((state) => state.label);

  const StatusIcon = connected ? IconCloudCheck : IconCloudOff;
  return (
    <>
      <StatusIcon
        color={connected ? "#0b0" : "#b00"}
        style={{
          transform: "translateY(-0.05em)",
          width: "1.2em",
          height: "1.2em",
        }}
      />
      <Box px="xs" sx={{ flexGrow: 1 }}>
        {label === "" ? server : label}
      </Box>
    </>
  );
}
