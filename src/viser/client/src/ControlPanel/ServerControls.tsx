import { ViewerContext } from "../App";
import * as THREE from "three";
import {
  Box,
  Button,
  Radio,
  Group,
  Divider,
  Select,
  Stack,
  Switch,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { IconHomeMove, IconPhoto, IconViewfinder } from "@tabler/icons-react";
import { Stats } from "@react-three/drei";
import React from "react";
import SceneTreeTable from "./SceneTreeTable";

export default function ServerControls() {
  const viewer = React.useContext(ViewerContext)!;
  const [showStats, setShowStats] = React.useState(false);

  function triggerBlur(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Enter") return;
    event.currentTarget.blur();
    event.currentTarget.focus();
  }
  const MemoizedTable = React.memo(SceneTreeTable);

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
        <Radio.Group
          onChange={(value) => {
            viewer!.useGui.setState({
              // Typing hack.
              cameraControlMode:
                value === "world-orbit" ? "world-orbit" : "camera-centric",
            });
          }}
          label="Camera"
          value={viewer!.useGui((state) => state.cameraControlMode)}
        >
          <Group mb="xs">
            <Tooltip
              label="Mouse and arrow keys will orbit around a look-at point in the world."
              maw={300}
              multiline
              refProp="rootRef"
            >
              <Radio value="world-orbit" label="Orbit" size="sm" />
            </Tooltip>
            <Tooltip
              label="Mouse and arrow keys will rotate the camera around itself."
              maw={300}
              multiline
              refProp="rootRef"
            >
              <Radio value="camera-centric" label="Camera-centric" size="sm" />
            </Tooltip>
          </Group>
        </Radio.Group>
        <Button
          onClick={() => {
            viewer.resetCameraViewRef.current!();
          }}
          leftSection={<IconViewfinder size="1rem" />}
        >
          Reset View
        </Button>
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
          <Text mb="0.2em" fw={500}>
            Scene tree
          </Text>
          <MemoizedTable />
        </Box>
      </Stack>
    </>
  );
}
