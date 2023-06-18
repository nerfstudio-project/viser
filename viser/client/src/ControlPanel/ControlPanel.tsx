import {
  IconAdjustments,
  IconBinaryTree2,
  IconCloudCheck,
  IconCloudOff,
  IconTool,
} from "@tabler/icons-react";
import { Tabs, TabsValue } from "@mantine/core";
import { ViewerContext } from "..";
import React from "react";
import GeneratedControls from "./Generated";
import ServerControls from "./Server";
import SceneTreeTable from "./SceneTreeTable";

/** Root component for control panel. Parents a set of control tabs. */
export default function ControlPanel() {
  const viewer = React.useContext(ViewerContext)!;
  const showGenerated = viewer.useGui((state) => state.guiNames.length > 0);

  const [tabState, setTabState] = React.useState<TabsValue>("server");

  // Switch to generated tab once populated.
  React.useEffect(() => {
    showGenerated && setTabState("generated");
  }, [showGenerated]);

  return (
    <Tabs radius="xs" value={tabState} onTabChange={setTabState}>
      <Tabs.List>
        {showGenerated ? (
          <Tabs.Tab value="generated" icon={<IconAdjustments size="0.8rem" />}>
            Control
          </Tabs.Tab>
        ) : null}
        <Tabs.Tab value="server" icon={<IconTool size="1rem" />}>
          Server
        </Tabs.Tab>
        <Tabs.Tab value="scene" icon={<IconBinaryTree2 size="1rem" />}>
          Scene
        </Tabs.Tab>
      </Tabs.List>

      {showGenerated ? (
        <Tabs.Panel value="generated" pt="xs" p="sm">
          <GeneratedControls />
        </Tabs.Panel>
      ) : null}

      <Tabs.Panel value="server" pt="xs" p="sm">
        <ServerControls />
      </Tabs.Panel>

      <Tabs.Panel value="scene" pt="xs" p="sm">
        <SceneTreeTable compact={true} />
      </Tabs.Panel>
    </Tabs>
  );
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
