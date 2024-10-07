import { ViewerContext } from "../App";
import {
  Box,
  Button,
  Divider,
  Group,
  Modal,
  NumberInput,
  Stack,
  Switch,
  Text,
  TextInput,
} from "@mantine/core";
import { IconCamera, IconHomeMove, IconPhoto } from "@tabler/icons-react";
import { Stats } from "@react-three/drei";
import React from "react";
import SceneTreeTable from "./SceneTreeTable";
import { useDisclosure } from "@mantine/hooks";
import { Vector3 } from "three";
import { computeT_threeworld_world } from "../WorldTransformUtils";

export default function ServerControls() {
  const viewer = React.useContext(ViewerContext)!;
  const [showStats, setShowStats] = React.useState(false);

  function triggerBlur(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Enter") return;
    event.currentTarget.blur();
    event.currentTarget.focus();
  }
  const MemoizedTable = React.memo(SceneTreeTable);
  const [setCameraModalOpened, { open: openSetCamera, close: closeSetCamera }] =
    useDisclosure(false);

  const cameraControls = viewer.cameraControlRef.current;

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
        <Button
          onClick={() => {
            openSetCamera();
          }}
          fullWidth
          leftSection={<IconCamera size="1rem" />}
          style={{ height: "1.875rem" }}
        >
          Set View
        </Button>
        <Modal
          opened={setCameraModalOpened}
          onClose={closeSetCamera}
          withCloseButton={true}
        >
          <Stack>
            <Text fw={500}>Camera Position</Text>
            <Group grow>
              <NumberInput
                label="X"
                placeholder="Enter X coordinate"
                decimalScale={2}
              />
              <NumberInput
                label="Y"
                placeholder="Enter Y coordinate"
                decimalScale={2}
              />
              <NumberInput
                label="Z"
                placeholder="Enter Z coordinate"
                decimalScale={2}
              />{" "}
            </Group>
            <Text fw={500}>Camera Target</Text>
            <Group grow>
              <NumberInput
                label="X"
                placeholder="Enter X coordinate"
                decimalScale={2}
              />
              <NumberInput
                label="Y"
                placeholder="Enter Y coordinate"
                decimalScale={2}
              />
              <NumberInput
                label="Z"
                placeholder="Enter Z coordinate"
                decimalScale={2}
              />
            </Group>
            <Switch label="Enable Transition" />
            <Button
              onClick={() => {
                const positionX = parseFloat(
                  (
                    document.querySelector(
                      'input[placeholder="Enter X coordinate"]',
                    ) as HTMLInputElement
                  ).value,
                );
                const positionY = parseFloat(
                  (
                    document.querySelector(
                      'input[placeholder="Enter Y coordinate"]',
                    ) as HTMLInputElement
                  ).value,
                );
                const positionZ = parseFloat(
                  (
                    document.querySelector(
                      'input[placeholder="Enter Z coordinate"]',
                    ) as HTMLInputElement
                  ).value,
                );
                const targetX = parseFloat(
                  (
                    document.querySelectorAll(
                      'input[placeholder="Enter X coordinate"]',
                    )[1] as HTMLInputElement
                  ).value,
                );
                const targetY = parseFloat(
                  (
                    document.querySelectorAll(
                      'input[placeholder="Enter Y coordinate"]',
                    )[1] as HTMLInputElement
                  ).value,
                );
                const targetZ = parseFloat(
                  (
                    document.querySelectorAll(
                      'input[placeholder="Enter Z coordinate"]',
                    )[1] as HTMLInputElement
                  ).value,
                );
                const enableTransition = (
                  document.querySelector(
                    'input[type="checkbox"]',
                  ) as HTMLInputElement
                ).checked;

                const T_threeworld_world = computeT_threeworld_world(viewer);

                const worldPosition = new Vector3(
                  positionX,
                  positionY,
                  positionZ,
                );
                const threeWorldPosition =
                  worldPosition.applyMatrix4(T_threeworld_world);

                const worldTarget = new Vector3(targetX, targetY, targetZ);
                const threeWorldTarget =
                  worldTarget.applyMatrix4(T_threeworld_world);

                cameraControls!.setPosition(
                  threeWorldPosition.x,
                  threeWorldPosition.y,
                  threeWorldPosition.z,
                  enableTransition,
                );
                cameraControls!.setTarget(
                  threeWorldTarget.x,
                  threeWorldTarget.y,
                  threeWorldTarget.z,
                  enableTransition,
                );
                closeSetCamera();
              }}
            >
              Apply Changes
            </Button>
          </Stack>
        </Modal>
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
