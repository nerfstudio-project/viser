import { ViewerContext } from "../App";
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
            const supportsFileSystemAccess =
              "showSaveFilePicker" in window &&
              (() => {
                try {
                  return window.self === window.top;
                } catch {
                  return false;
                }
              })();

            if (supportsFileSystemAccess) {
              // File System Access API is supported. (eg Chrome)
              const fileHandlePromise = window.showSaveFilePicker({
                suggestedName: "render.png",
                types: [
                  {
                    accept: { "image/png": [".png"] },
                  },
                ],
              });
              viewer.canvasRef.current?.toBlob(async (blob) => {
                if (blob === null) {
                  console.error("Export failed");
                  return;
                }

                const handle = await fileHandlePromise;
                const writableStream = await handle.createWritable();
                await writableStream.write(blob);
                await writableStream.close();
              });
            } else {
              // File System Access API is not supported. (eg Firefox)
              viewer.canvasRef.current?.toBlob((blob) => {
                if (blob === null) {
                  console.error("Export failed");
                  return;
                }
                const href = URL.createObjectURL(blob);

                // Download a file by creating a link and then clicking it.
                const link = document.createElement("a");
                link.href = href;
                const filename = "render.png";
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(href);
              });
            }
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
