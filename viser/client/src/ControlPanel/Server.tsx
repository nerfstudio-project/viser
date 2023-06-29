import { ViewerContext } from "..";
import { isTexture } from "../WebsocketInterface";
import { Button, Stack, Switch, TextInput } from "@mantine/core";
import { Stats } from "@react-three/drei";
import { IconPhoto } from "@tabler/icons-react";
import React from "react";

export default function ServerControls() {
  const viewer = React.useContext(ViewerContext)!;
  const [showStats, setShowStats] = React.useState(false);

  function triggerBlur(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Enter") return;
    event.currentTarget.blur();
    event.currentTarget.focus();
  }

  return (
    <>
      {showStats ? <Stats className="stats-panel" /> : null}
      <Stack spacing="xs">
        <TextInput
          label="Label"
          defaultValue={viewer.useGui((state) => state.label)}
          onBlur={(event) =>
            viewer.useGui.setState({ label: event.currentTarget.value })
          }
          onKeyDown={triggerBlur}
        />
        <TextInput
          label="Server"
          defaultValue={viewer.useGui((state) => state.server)}
          onBlur={(event) =>
            viewer.useGui.setState({ server: event.currentTarget.value })
          }
          onKeyDown={triggerBlur}
        />
        <Button
          onClick={() => {
            if (!isTexture(viewer.sceneRef.current?.background)) {
              // This should never happen.
              alert("No background to download!");
              return;
            }

            const data = viewer.sceneRef.current?.background.image.src;
            const link = document.createElement("a");
            link.download = "background";
            link.href = data;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          }}
          fullWidth
          disabled={!viewer.useGui((state) => state.backgroundAvailable)}
          leftIcon={<IconPhoto size="1rem" />}
        >
          Download Background
        </Button>
        <Switch
          label="WebGL Statistics"
          onChange={(event) => {
            setShowStats(event.currentTarget.checked);
          }}
        />
      </Stack>
    </>
  );
}
