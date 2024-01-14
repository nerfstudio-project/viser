import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import GeneratedGuiContainer from "./Generated";
import { ViewerContext } from "../App";

import ServerControls from "./ServerControls";
import {
  ActionIcon,
  Anchor,
  Box,
  Button,
  Collapse,
  CopyButton,
  Flex,
  Loader,
  Modal,
  Stack,
  Text,
  TextInput,
  Tooltip,
  Transition,
  useMantineTheme,
} from "@mantine/core";
import {
  IconAdjustments,
  IconCloudCheck,
  IconArrowBack,
  IconShare,
  IconCopy,
  IconCheck,
  IconPlugConnectedX,
} from "@tabler/icons-react";
import React from "react";
import BottomPanel from "./BottomPanel";
import FloatingPanel from "./FloatingPanel";
import { ThemeConfigurationMessage } from "../WebsocketMessages";
import SidebarPanel from "./SidebarPanel";
import { sendWebsocketMessage } from "../WebsocketFunctions";

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
            <ShareButton />
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
            <ShareButton />
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
          <ShareButton />
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

function ShareButton() {
  const viewer = React.useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketConnected);
  const shareUrl = viewer.useGui((state) => state.shareUrl);
  const setShareUrl = viewer.useGui((state) => state.setShareUrl);

  const [doingSomething, setDoingSomething] = React.useState(false);

  const [shareModalOpened, { open: openShareModal, close: closeShareModal }] =
    useDisclosure(false);

  // Turn off loader when share URL is set.
  React.useEffect(() => {
    if (shareUrl !== null) {
      setDoingSomething(false);
    }
  }, [shareUrl]);
  React.useEffect(() => {
    if (!connected && shareModalOpened) closeShareModal();
  }, [connected, shareModalOpened]);

  if (viewer.useGui((state) => state.theme).show_share_button === false)
    return null;

  return (
    <>
      <Tooltip
        zIndex={100}
        label={connected ? "Share" : "Share (needs connection)"}
        withinPortal
      >
        <div>
          <ActionIcon
            onClick={(evt) => {
              evt.stopPropagation();
              openShareModal();
            }}
            disabled={!connected}
          >
            <IconShare stroke={2} height="1.125em" width="1.125em" />
          </ActionIcon>
        </div>
      </Tooltip>
      <Modal
        title="Share"
        opened={shareModalOpened}
        onClose={closeShareModal}
        withCloseButton={false}
        zIndex={100}
        withinPortal
        onClick={(evt) => evt.stopPropagation()}
        onMouseDown={(evt) => evt.stopPropagation()}
        onMouseMove={(evt) => evt.stopPropagation()}
        onMouseUp={(evt) => evt.stopPropagation()}
        styles={{ title: { fontWeight: 600 } }}
      >
        {shareUrl === null ? (
          <>
            {/*<Select
                label="Server"
                data={["viser-us-west (https://share.viser.studio)"]}
                withinPortal
                {...form.getInputProps("server")}
              /> */}
            {doingSomething ? (
              <Stack mb="xl">
                <Loader size="xl" mx="auto" />
              </Stack>
            ) : (
              <Stack mb="md">
                <Text>
                  Create a public, shareable URL to this Viser instance.
                </Text>
                <Button
                  fullWidth
                  onClick={() => {
                    sendWebsocketMessage(viewer.websocketRef, {
                      type: "ShareUrlRequest",
                    });
                    setDoingSomething(true); // Loader state will help with debouncing.
                  }}
                >
                  Request Share URL
                </Button>
              </Stack>
            )}
          </>
        ) : (
          <>
            <Text>Share URL is connected!</Text>
            <Stack my="md">
              <Flex justify="space-between" columnGap="0.5em" align="center">
                <TextInput value={shareUrl} style={{ flexGrow: "1" }} />
                <Tooltip zIndex={100} label="Copy" withinPortal>
                  <CopyButton value={shareUrl}>
                    {({ copied, copy }) => (
                      <ActionIcon size="lg" onClick={copy}>
                        {copied ? <IconCheck /> : <IconCopy />}
                      </ActionIcon>
                    )}
                  </CopyButton>
                </Tooltip>
              </Flex>
              <Button
                leftIcon={
                  <IconPlugConnectedX height="1.375em" width="1.375em" />
                }
                color="red"
                onClick={() => {
                  sendWebsocketMessage(viewer.websocketRef, {
                    type: "ShareUrlDisconnect",
                  });
                  setShareUrl(null);
                }}
              >
                Disconnect
              </Button>
            </Stack>
          </>
        )}
        <Text size="xs">
          Not working? Consider{" "}
          <Anchor href="https://github.com/nerfstudio-project/viser/issues">
            filing an issue
          </Anchor>{" "}
          on GitHub.
        </Text>
      </Modal>
    </>
  );
}
