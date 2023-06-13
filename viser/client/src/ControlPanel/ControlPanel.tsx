import {
  IconAdjustments,
  IconCloudCheck,
  IconCloudOff,
  IconLink,
} from "@tabler/icons-react";
import { Tabs } from "@mantine/core";
import { ViewerContext } from "..";
import FloatingPanel from "./FloatingPanel";
import React from "react";
import GeneratedControls from "./Generated";
import ServerControls from "./Server";

/** Root component for control panel. Parents a set of control tabs. */
export default function ControlPanel() {
  return (
    <FloatingPanel>
      <FloatingPanel.Handle>
        <ConnectionStatus />
      </FloatingPanel.Handle>
      <FloatingPanel.Contents>
        <Tabs radius="xs" defaultValue="generated">
          <Tabs.List>
            <Tabs.Tab
              value="generated"
              icon={<IconAdjustments size="0.8rem" />}
            >
              Control
            </Tabs.Tab>
            <Tabs.Tab value="server" icon={<IconLink size="1rem" />}>
              Server
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="generated" pt="xs">
            <GeneratedControls />
          </Tabs.Panel>

          <Tabs.Panel value="server" pt="xs">
            <ServerControls />
          </Tabs.Panel>
        </Tabs>
      </FloatingPanel.Contents>
    </FloatingPanel>
  );
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
