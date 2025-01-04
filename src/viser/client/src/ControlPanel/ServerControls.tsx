import { ViewerContext } from "../App";
import {
  Box,
  Button,
  Divider,
  Stack,
  Switch,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { IconHomeMove, IconPhoto } from "@tabler/icons-react";
import { Stats } from "@react-three/drei";
import React from "react";
import SceneTreeTable from "./SceneTreeTable";

const MemoizedTable = React.memo(SceneTreeTable);

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
      <Stack gap="xs">
        <TextInput
          label="Server"
          defaultValue={viewer.useGui((state) => state.server)}
          onBlur={(event) =>
            viewer.useGui.setState({ server: event.currentTarget.value })
          }
          onKeyDown={triggerBlur}
          styles={{
            input: {
              minHeight: "1.75rem",
              height: "1.75rem",
              padding: "0 0.5em",
            },
          }}
        />
        <TextInput
          label="Label"
          defaultValue={viewer.useGui((state) => state.label)}
          onBlur={(event) =>
            viewer.useGui.setState({ label: event.currentTarget.value })
          }
          onKeyDown={triggerBlur}
          styles={{
            input: {
              minHeight: "1.75rem",
              height: "1.75rem",
              padding: "0 0.5em",
            },
          }}
          mb="0.375em"
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
          leftSection={<IconPhoto size="1rem" />}
          style={{ height: "1.875rem" }}
        >
          Export Canvas
        </Button>
        <Button
          onClick={() => {
            viewer.resetCameraViewRef.current!();
          }}
          fullWidth
          leftSection={<IconHomeMove size="1rem" />}
          style={{ height: "1.875rem" }}
        >
          Reset View
        </Button>
        <Tooltip
          label={
            <>
              Show tool for setting the look-at point and
              <br />
              up direction of the camera.
              <br />
              <br />
              These can be used to re-orient the camera&apos;s
              <br />
              orbit controls.
            </>
          }
          refProp="rootRef"
          position="top-start"
        >
          <Switch
            radius="sm"
            label="Orbit Origin Tool"
            onChange={(event) => {
              viewer.useGui.setState({
                showOrbitOriginTool: event.currentTarget.checked,
              });
            }}
            size="sm"
          />
        </Tooltip>
        <Switch
          radius="sm"
          label="WebGL Statistics"
          onChange={(event) => {
            setShowStats(event.currentTarget.checked);
          }}
          size="sm"
        />
        <Divider mt="xs" />
        <Box>
          <Text mb="0.2em" fw={500} fz="sm">
            Scene tree
          </Text>
          <MemoizedTable />
        </Box>
      </Stack>
    </>
  );
}
