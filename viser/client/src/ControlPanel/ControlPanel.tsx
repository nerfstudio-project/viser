import { useDisclosure } from "@mantine/hooks";
import { ViewerContext } from "..";
import GeneratedGuiContainer from "./Generated";
import SceneTreeTable from "./SceneTreeTable";
import ServerControls from "./Server";
import {
  ActionIcon,
  Box,
  Button,
  CloseButton,
  Collapse,
  Paper,
  Tabs,
  TabsValue,
  Tooltip,
} from "@mantine/core";
import {
  IconAdjustments,
  IconX,
  IconBinaryTree2,
  IconCloudCheck,
  IconCloudOff,
  IconSettings,
  IconTool,
  IconArrowBack,
} from "@tabler/icons-react";
import React from "react";

// Must match constant in Python.
const ROOT_CONTAINER_ID = "root";

/** Root component for control panel. Parents a set of control tabs. */
export default function ControlPanel() {
  const viewer = React.useContext(ViewerContext)!;

  // TODO: will result in unnecessary re-renders
  const showGenerated =
    Object.keys(viewer.useGui((state) => state.guiConfigFromId)).length > 0;

  const [tabState, setTabState] = React.useState<TabsValue>("server");

  // Switch to generated tab once populated.
  React.useEffect(() => {
    showGenerated && setTabState("generated");
  }, [showGenerated]);

  const MemoizedTable = React.memo(SceneTreeTable);

  const [showSettings, { toggle }] = useDisclosure(false);

  if (!showGenerated) {
    return (
      <Box p="sm">
        <ServerControls />
      </Box>
    );
  } else {
    return (
      <>
        <ActionIcon
          onClick={toggle}
          sx={{
            position: "absolute",
            right: "0.25em",
            top: "0.375em",
          }}
        >
          <Tooltip
            label={showSettings ? "Return to GUI" : "Connection & diagnostics"}
          >
            {showSettings ? <IconArrowBack /> : <IconAdjustments />}
          </Tooltip>
        </ActionIcon>
        <Collapse in={!showSettings}>
          <GeneratedGuiContainer containerId={ROOT_CONTAINER_ID} />
        </Collapse>
        <Collapse p="sm" in={showSettings}>
          <ServerControls />
        </Collapse>
      </>
    );
    // <ServerControls />
  }
}
/* Icon and label telling us the current status of the websocket connection. */
export function ConnectionStatus() {
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
