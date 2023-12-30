import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import GeneratedGuiContainer from "./Generated";
import { ViewerContext } from "../App";
import ServerControls from "./ServerControls";
import {
  ActionIcon,
  Box,
  Collapse,
  Loader,
  Tooltip,
  Transition,
  useMantineTheme,
} from "@mantine/core";
import {
  IconAdjustments,
  IconCloudCheck,
  IconArrowBack,
} from "@tabler/icons-react";
import React from "react";
import BottomPanel from "./BottomPanel";
import FloatingPanel from "./FloatingPanel";
import { ThemeConfigurationMessage } from "../WebsocketMessages";
import SidebarPanel from "./SidebarPanel";
import CameraTrajectoryPanel from "./CameraTrajectoryPanel";

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

  const controlWidthString = viewer.useGui(
    (state) => state.theme.control_width,
  );
  const controlWidth = (
    controlWidthString == "small"
      ? "16em"
      : controlWidthString == "medium"
      ? "20em"
      : controlWidthString == "large"
      ? "24em"
      : null
  )!;

  const generatedServerToggleButton = (
    <Box sx={{ display: showGenerated ? undefined : "none" }}>
      <ActionIcon
        onClick={(evt) => {
          evt.stopPropagation();
          toggle();
        }}
      >
        <Tooltip
          zIndex={100}
          label={showSettings ? "Return to GUI" : "Connection & diagnostics"}
          withinPortal
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
      <Collapse in={!showGenerated || showSettings} p="xs" pt="0.375em">
        <CameraTrajectoryPanel />
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
          <BottomPanel.HideWhenCollapsed>
            {generatedServerToggleButton}
          </BottomPanel.HideWhenCollapsed>
        </BottomPanel.Handle>
        <BottomPanel.Contents>{panelContents}</BottomPanel.Contents>
      </BottomPanel>
    );
  } else if (props.control_layout === "floating") {
    /* Floating layout. */
    return (
      <FloatingPanel width={controlWidth}>
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
      <SidebarPanel
        width={controlWidth}
        collapsible={props.control_layout === "collapsible"}
      >
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
  const label = useGui((state) => state.label);

  return (
    <>
      <div style={{ width: "1.1em" }} /> {/* Spacer. */}
      <Transition transition="skew-down" mounted={connected}>
        {(styles) => (
          <IconCloudCheck
            color={"#0b0"}
            style={{
              position: "absolute",
              width: "1.25em",
              height: "1.25em",
              ...styles,
            }}
          />
        )}
      </Transition>
      <Transition transition="skew-down" mounted={!connected}>
        {(styles) => (
          <Loader
            size="xs"
            variant="bars"
            color="red"
            style={{ position: "absolute", ...styles }}
          />
        )}
      </Transition>
      <Box px="xs" sx={{ flexGrow: 1 }} lts={"-0.5px"} pt="0.1em">
        {label !== "" ? label : connected ? "Connected" : "Connecting..."}
      </Box>
    </>
  );
}
