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

              const supportsFileSystemAccess =
                'showSaveFilePicker' in window &&
                (() => {
                  try {
                    return window.self === window.top;
                  } catch {
                    return false;
                  }
                })();
              // If the File System Access API is supported…
              if (supportsFileSystemAccess) {
                let handle = null;
                try {
                  handle = await window.showSaveFilePicker({
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
              }

              const href = URL.createObjectURL(blob);

              // create "a" HTML element with href to file
              const link = document.createElement('a');
              link.href = href;

              const filename = 'render.png';
              link.download = filename;
              document.body.appendChild(link);
              link.click();
              // clean up "a" element & remove ObjectURL
              document.body.removeChild(link);
              URL.revokeObjectURL(href);
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
