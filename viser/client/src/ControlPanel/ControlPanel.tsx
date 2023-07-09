import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import { ViewerContext } from "..";
import GeneratedGuiContainer from "./Generated";
import ServerControls from "./Server";
import {
  ActionIcon,
  Aside,
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

// Must match constant in Python.
const ROOT_CONTAINER_ID = "root";

export default function ControlPanel(props: { fixed_sidebar: boolean }) {
  const theme = useMantineTheme();
  const useMobileView = useMediaQuery(`(max-width: ${theme.breakpoints.xs})`);

  // TODO: will result in unnecessary re-renders
  const viewer = React.useContext(ViewerContext)!;
  const showGenerated =
    Object.keys(viewer.useGui((state) => state.guiConfigFromId)).length > 0;
  const [showSettings, { toggle }] = useDisclosure(false);
  const handleContents = (
    <>
      <ConnectionStatus />
      {/* We can't apply translateY directly to the ActionIcon, since it's used by
      Mantine for the active/click indicator. */}
      <Box
        sx={{
          position: "absolute",
          right: "0.5em",
          top: "50%",
          transform: "translateY(-50%)",
          display: showGenerated ? undefined : "none",
        }}
      >
        <ActionIcon
          onClick={(evt) => {
            evt.stopPropagation();
            toggle();
          }}
        >
          <Tooltip
            label={showSettings ? "Return to GUI" : "Connection & diagnostics"}
          >
            {showSettings ? <IconArrowBack /> : <IconAdjustments />}
          </Tooltip>
        </ActionIcon>
      </Box>
    </>
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
    return (
      <BottomPanel>
        <BottomPanel.Handle>{handleContents}</BottomPanel.Handle>
        <BottomPanel.Contents>{panelContents}</BottomPanel.Contents>
      </BottomPanel>
    );
  } else if (props.fixed_sidebar) {
    return (
      <Aside
        hiddenBreakpoint={"xs"}
        sx={(theme) => ({
          width: "20em",
          boxSizing: "border-box",
          right: 0,
          position: "absolute",
          top: "0em",
          bottom: "0em",
          borderLeft: "1px solid",
          borderColor:
            theme.colorScheme == "light"
              ? theme.colors.gray[4]
              : theme.colors.dark[4],
        })}
      >
        <Box
          p="sm"
          sx={(theme) => ({
            backgroundColor:
              theme.colorScheme == "dark"
                ? theme.colors.dark[5]
                : theme.colors.gray[1],
            lineHeight: "1.5em",
            fontWeight: 400,
          })}
        >
          {handleContents}
        </Box>
        {panelContents}
      </Aside>
    );
  } else {
    return (
      <FloatingPanel>
        <FloatingPanel.Handle>{handleContents}</FloatingPanel.Handle>
        <FloatingPanel.Contents>{panelContents}</FloatingPanel.Contents>
      </FloatingPanel>
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
          transform: "translateY(0.1em) scale(1.2)",
          width: "1em",
          height: "1em",
        }}
      />
      &nbsp; &nbsp;
      {label === "" ? server : label}
    </>
  );
}
