import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import GeneratedGuiContainer from "./Generated";
import { ViewerContext } from "../App";
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
  IconChevronLeft,
  IconChevronRight,
} from "@tabler/icons-react";
import React from "react";
import BottomPanel from "./BottomPanel";
import FloatingPanel, { FloatingPanelContext } from "./FloatingPanel";
import { ThemeConfigurationMessage } from "../WebsocketMessages";

// Must match constant in Python.
const ROOT_CONTAINER_ID = "root";

/** Hides contents when floating panel is collapsed. */
function HideWhenCollapsed({ children }: { children: React.ReactNode }) {
  const expanded = React.useContext(FloatingPanelContext)?.expanded ?? true;
  return expanded ? children : null;
}

export default function ControlPanel(props: {
  control_layout: ThemeConfigurationMessage["control_layout"];
}) {
  const theme = useMantineTheme();
  const useMobileView = useMediaQuery(`(max-width: ${theme.breakpoints.xs})`);
  
  // TODO: will result in unnecessary re-renders
  const viewer = React.useContext(ViewerContext)!;
  const showGenerated = viewer.useGui(
    (state) => "root" in state.guiIdSetFromContainerId,
  );
  const [showSettings, { toggle }] = useDisclosure(false);
  const [collapsed, { toggle: toggleCollapse }] = useDisclosure(false);
  const handleContents = (
    <>
      <ConnectionStatus />
      <HideWhenCollapsed>
        {/* We can't apply translateY directly to the ActionIcon, since it's used by
      Mantine for the active/click indicator. */}
        <Box
          sx={{
            position: "absolute",
            right: props.control_layout === "collapsible" ? "2.5em" : "0.5em",
            top: "50%",
            transform: "translateY(-50%)",
            display: showGenerated ? undefined : "none",
            zIndex: 100,
          }}
        >
          <ActionIcon
            onClick={(evt) => {
              evt.stopPropagation();
              toggle();
            }}
          >
            <Tooltip
              label={
                showSettings ? "Return to GUI" : "Connection & diagnostics"
              }
            >
              {showSettings ? <IconArrowBack /> : <IconAdjustments />}
            </Tooltip>
          </ActionIcon>
        </Box>
      </HideWhenCollapsed>
      <Box
        sx={{
          position: "absolute",
          right: "0.5em",
          top: "50%",
          transform: "translateY(-50%)",
          display:
            props.control_layout === "collapsible" && !useMobileView
              ? undefined
              : "none",
          zIndex: 100,
        }}
      >
        <ActionIcon
          onClick={(evt) => {
            evt.stopPropagation();
            toggleCollapse();
          }}
        >
          <Tooltip label={"Collapse Sidebar"}>{<IconChevronRight />}</Tooltip>
        </ActionIcon>
      </Box>
    </>
  );

  const collapsedView = (
    <div
      style={{
        borderTopLeftRadius: "15%",
        borderBottomLeftRadius: "15%",
        borderTopRightRadius: 0,
        borderBottomRightRadius: 0,
        backgroundColor:
          theme.colorScheme == "dark"
            ? theme.colors.dark[5]
            : theme.colors.gray[2],
        padding: "0.5em",
      }}
    >
      <ActionIcon
        onClick={(evt) => {
          evt.stopPropagation();
          toggleCollapse();
        }}
      >
        <Tooltip label={"Show Sidebar"}>{<IconChevronLeft />}</Tooltip>
      </ActionIcon>
    </div>
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
  } else if (props.control_layout !== "floating") {
    return (
      <>
        <Box
          sx={{
            position: "absolute",
            right: collapsed ? "0em" : "-2.5em",
            top: "0.5em",
            transitionProperty: "right",
            transitionDuration: "0.5s",
            transitionDelay: "0.25s",
          }}
        >
          {collapsedView}
        </Box>
        <Aside
          hiddenBreakpoint={"xs"}
          fixed
          sx={(theme) => ({
            width: collapsed ? 0 : "20em",
            bottom: 0,
            overflow: "scroll",
            boxSizing: "border-box",
            borderLeft: "1px solid",
            borderColor:
              theme.colorScheme == "light"
                ? theme.colors.gray[4]
                : theme.colors.dark[4],
            transition: "width 0.5s 0s",
          })}
        >
          <Box
            sx={() => ({
              width: "20em",
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
                position: "relative",
                zIndex: 1,
              })}
            >
              {handleContents}
            </Box>
            {panelContents}
          </Box>
        </Aside>
      </>
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
    <span
      style={{ display: "flex", alignItems: "center", width: "max-content" }}
    >
      <StatusIcon
        color={connected ? "#0b0" : "#b00"}
        style={{
          transform: "translateY(-0.05em)",
          width: "1.2em",
          height: "1.2em",
        }}
      />
      &nbsp; &nbsp;
      {label === "" ? server : label}
    </span>
  );
}
