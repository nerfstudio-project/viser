import { ViewerContext } from "..";
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
          onClick={async () => {
            viewer.canvasRef.current?.toBlob(async (blob) => {
              if (blob === null) {
                console.error("Render failed!");
                return;
              }

              let handle = null;
              try {
                handle = await showSaveFilePicker({
                  suggestedName: "render.png",
                  types: [
                    {
                      accept: { "image/png": [".png"] },
                    },
                  ],
                });
              } catch (e) {
                console.log(e);
              }

              if (handle) {
                const writableStream = await handle.createWritable();
                await writableStream.write(blob);
                await writableStream.close();
              }
            });
          }}
          fullWidth
          leftIcon={<IconPhoto size="1rem" />}
        >
          Export Canvas
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
