import { decodeAsync, decode } from "@msgpack/msgpack";
import { Message } from "./WebsocketMessages";
import { decompress } from "fflate";

import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { ViewerContext } from "./ViewerContext";
import {
  ActionIcon,
  NumberInput,
  Paper,
  Progress,
  Select,
  Slider,
  Tooltip,
  useMantineTheme,
} from "@mantine/core";
import {
  IconPlayerPauseFilled,
  IconPlayerPlayFilled,
} from "@tabler/icons-react";

/** Download, decompress, and deserialize a file, which should be serialized
 * via msgpack and compressed via gzip. Also takes a hook for status updates. */
async function deserializeGzippedMsgpackFile<T>(
  fileUrl: string,
  setStatus: (status: { downloaded: number; total: number }) => void,
): Promise<T> {
  const response = await fetch(fileUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch the file: ${response.statusText}`);
  }
  return new Promise<T>((resolve) => {
    const gzipTotalLength = parseInt(response.headers.get("Content-Length")!);
    if (typeof DecompressionStream === "undefined") {
      // Implementation without DecompressionStream.
      console.log("DecompressionStream is unavailable. Using fallback.");
      setStatus({ downloaded: 0.1 * gzipTotalLength, total: gzipTotalLength });
      response.arrayBuffer().then((buffer) => {
        setStatus({
          downloaded: 0.8 * gzipTotalLength,
          total: gzipTotalLength,
        });
        decompress(new Uint8Array(buffer), (error, result) => {
          setStatus({
            downloaded: 1.0 * gzipTotalLength,
            total: gzipTotalLength,
          });
          resolve(decode(result) as T);
        });
      });
    } else {
      // Stream: fetch -> gzip -> msgpack.
      let gzipReceived = 0;
      const progressStream = // Count number of (compressed) bytes.
        new TransformStream({
          transform(chunk, controller) {
            gzipReceived += chunk.length;
            setStatus({ downloaded: gzipReceived, total: gzipTotalLength });
            controller.enqueue(chunk);
          },
        });
      decodeAsync(
        response
          .body!.pipeThrough(progressStream)
          .pipeThrough(new DecompressionStream("gzip")),
      ).then((val) => resolve(val as T));
    }
  });
}

/** Deserialize embedded base64-encoded gzip-compressed msgpack data.
 * Used for static embedding where scene data is inlined in the HTML. */
async function deserializeEmbeddedData<T>(
  base64Data: string,
  setStatus: (status: { downloaded: number; total: number }) => void,
): Promise<T> {
  // Decode base64 to Uint8Array.
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  setStatus({ downloaded: 0.5, total: 1.0 });

  // Decompress and decode msgpack.
  return new Promise<T>((resolve) => {
    decompress(bytes, (error, result) => {
      if (error) {
        console.error("Failed to decompress embedded data:", error);
      }
      setStatus({ downloaded: 1.0, total: 1.0 });
      resolve(decode(result) as T);
    });
  });
}

export interface SerializedMessages {
  durationSeconds: number;
  messages: [number, Message][]; // (time in seconds, message).
  viserVersion: string;
}

export function PlaybackFromFile({ fileUrl }: { fileUrl: string }) {
  const viewer = useContext(ViewerContext)!;
  const viewerMutable = viewer.mutable.current; // Get mutable once

  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const [status, setStatus] = useState({ downloaded: 0.0, total: 0.0 });
  const [playbackSpeed, setPlaybackSpeed] = useState("1x");
  const [paused, setPaused] = useState(false);
  const [recording, setRecording] = useState<SerializedMessages | null>(null);

  // Instead of removing all of the existing scene nodes, we're just going to hide them.
  // This will prevent unnecessary remounting when messages are looped.
  function resetScene() {
    const sceneTreeState = viewer.useSceneTree.getState();
    Object.keys(sceneTreeState).forEach((key) => {
      if (key === "") return;
      const node = sceneTreeState[key];
      const nodeMessage = node?.message;
      if (
        nodeMessage !== undefined &&
        (nodeMessage.type !== "FrameMessage" || nodeMessage.props.show_axes)
      ) {
        // ^ We don't hide intermediate frames. These can be created
        // automatically by addSceneNodeMakerParents(), in which case there
        // will be no message to un-hide them.
        viewer.sceneTreeActions.updateNodeAttributes(key, {
          visibility: false,
          wxyz: [1, 0, 0, 0],
          position: [0, 0, 0],
        });
      } else if (node !== undefined) {
        // Still reset poses for frames.
        viewer.sceneTreeActions.updateNodeAttributes(key, {
          wxyz: [1, 0, 0, 0],
          position: [0, 0, 0],
        });
      }
    });
  }

  const [currentTime, setCurrentTime] = useState(0.0);

  const theme = useMantineTheme();

  useEffect(() => {
    deserializeGzippedMsgpackFile<SerializedMessages>(fileUrl, setStatus).then(
      (data) => {
        console.log(
          "File loaded! Saved with Viser version:",
          data.viserVersion,
        );
        setRecording(data);
      },
    );
  }, []);

  const playbackMutable = useRef({ currentTime: 0.0, currentIndex: 0 });

  const updatePlayback = useCallback(() => {
    if (recording === null) return;
    const mutable = playbackMutable.current;

    // We have messages with times: [0.0, 0.01, 0.01, 0.02, 0.03]
    // We have our current time: 0.02
    // We want to get of a slice of all message _until_ the current time.
    if (mutable.currentIndex == 0) {
      // Reset the scene if sending the first message.
      resetScene();
    }
    for (
      ;
      mutable.currentIndex < recording.messages.length &&
      recording.messages[mutable.currentIndex][0] <= mutable.currentTime;
      mutable.currentIndex++
    ) {
      const message = recording.messages[mutable.currentIndex][1];
      viewerMutable.messageQueue.push(message);
    }

    // Don't loop for static scenes (durationSeconds === 0).
    if (
      mutable.currentTime >= recording.durationSeconds &&
      recording.durationSeconds > 0
    ) {
      mutable.currentIndex = 0;
      mutable.currentTime = recording.messages[0][0];
    }
    setCurrentTime(mutable.currentTime);
  }, [recording]);

  useEffect(() => {
    const playbackMultiplier = parseFloat(playbackSpeed); // '0.5x' -> 0.5
    if (recording !== null && !paused) {
      let lastUpdate = Date.now();
      const interval = setInterval(() => {
        const now = Date.now();
        playbackMutable.current.currentTime +=
          ((now - lastUpdate) / 1000.0) * playbackMultiplier;
        lastUpdate = now;

        updatePlayback();
        // Stop playback for static scenes once all messages are processed.
        if (
          playbackMutable.current.currentIndex === recording.messages.length &&
          recording.durationSeconds === 0.0
        ) {
          clearInterval(interval);
        }
      }, 1000.0 / 120.0);
      return () => clearInterval(interval);
    }
  }, [
    updatePlayback,
    recording,
    paused,
    playbackSpeed,
    viewerMutable.messageQueue,
    setCurrentTime,
  ]);

  // Pause/play with spacebar.
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.code === "Space") {
        setPaused(!paused);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [paused]); // Empty dependency array ensures this runs once on mount and cleanup on unmount

  const updateCurrentTime = useCallback(
    (value: number) => {
      if (value < playbackMutable.current.currentTime) {
        // Going backwards is more expensive...
        resetScene();
        playbackMutable.current.currentIndex = 0;
      }
      playbackMutable.current.currentTime = value;
      setCurrentTime(value);
      setPaused(true);
      updatePlayback();
    },
    [recording],
  );

  if (recording === null) {
    return (
      <div
        style={{
          position: "fixed",
          zIndex: 1,
          top: 0,
          bottom: 0,
          left: 0,
          right: 0,
          backgroundColor: darkMode ? theme.colors.dark[9] : "#fff",
        }}
      >
        <Progress
          value={(status.downloaded / status.total) * 100.0}
          radius={0}
          transitionDuration={0}
        />
      </div>
    );
  } else {
    return (
      <Paper
        radius="xs"
        shadow="0.1em 0 1em 0 rgba(0,0,0,0.1)"
        style={{
          position: "fixed",
          bottom: "1em",
          left: "50%",
          transform: "translateX(-50%)",
          width: "25em",
          maxWidth: "95%",
          zIndex: 1,
          padding: "0.5em",
          display: recording.durationSeconds === 0.0 ? "none" : "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.375em",
        }}
      >
        <ActionIcon
          size="md"
          variant="subtle"
          onClick={() => setPaused(!paused)}
        >
          {paused ? (
            <IconPlayerPlayFilled height="1.125em" width="1.125em" />
          ) : (
            <IconPlayerPauseFilled height="1.125em" width="1.125em" />
          )}
        </ActionIcon>
        <NumberInput
          size="xs"
          hideControls
          value={currentTime.toFixed(1)}
          step={0.01}
          styles={{
            wrapper: {
              width: "3.1em",
            },
            input: {
              padding: "0.2em",
              fontFamily: theme.fontFamilyMonospace,
              textAlign: "center",
            },
          }}
          onChange={(value) =>
            updateCurrentTime(
              typeof value === "number" ? value : parseFloat(value),
            )
          }
        />
        <Slider
          thumbSize={0}
          radius="xs"
          step={1e-4}
          style={{ flexGrow: 1 }}
          min={0}
          max={recording.durationSeconds}
          value={currentTime}
          onChange={updateCurrentTime}
          styles={{ thumb: { display: "none" } }}
        />
        <Tooltip zIndex={10} label={"Playback speed"} withinPortal>
          <Select
            size="xs"
            value={playbackSpeed}
            onChange={(val) => (val === null ? null : setPlaybackSpeed(val))}
            radius="xs"
            data={["0.5x", "1x", "2x", "4x", "8x"]}
            styles={{
              wrapper: { width: "3.25em" },
            }}
            comboboxProps={{ zIndex: 5, width: "5.25em" }}
          />
        </Tooltip>
      </Paper>
    );
  }
}

/** Component for playback from embedded base64 scene data.
 * Used for static embedding in HTML pages (e.g., myst-nb documentation). */
export function PlaybackFromEmbedData({
  base64Data,
}: {
  base64Data: string;
}) {
  const viewer = useContext(ViewerContext)!;
  const viewerMutable = viewer.mutable.current;

  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const [status, setStatus] = useState({ downloaded: 0.0, total: 0.0 });
  const [playbackSpeed, setPlaybackSpeed] = useState("1x");
  const [paused, setPaused] = useState(false);
  const [recording, setRecording] = useState<SerializedMessages | null>(null);

  function resetScene() {
    const sceneTreeState = viewer.useSceneTree.getState();
    Object.keys(sceneTreeState).forEach((key) => {
      if (key === "") return;
      const node = sceneTreeState[key];
      const nodeMessage = node?.message;
      if (
        nodeMessage !== undefined &&
        (nodeMessage.type !== "FrameMessage" || nodeMessage.props.show_axes)
      ) {
        viewer.sceneTreeActions.updateNodeAttributes(key, {
          visibility: false,
          wxyz: [1, 0, 0, 0],
          position: [0, 0, 0],
        });
      } else if (node !== undefined) {
        viewer.sceneTreeActions.updateNodeAttributes(key, {
          wxyz: [1, 0, 0, 0],
          position: [0, 0, 0],
        });
      }
    });
  }

  const [currentTime, setCurrentTime] = useState(0.0);
  const theme = useMantineTheme();

  useEffect(() => {
    deserializeEmbeddedData<SerializedMessages>(base64Data, setStatus).then(
      (data) => {
        console.log(
          "Embedded data loaded! Saved with Viser version:",
          data.viserVersion,
        );
        setRecording(data);
      },
    );
  }, [base64Data]);

  const playbackMutable = useRef({ currentTime: 0.0, currentIndex: 0 });

  const updatePlayback = useCallback(() => {
    if (recording === null) return;
    const mutable = playbackMutable.current;

    if (mutable.currentIndex == 0) {
      resetScene();
    }
    for (
      ;
      mutable.currentIndex < recording.messages.length &&
      recording.messages[mutable.currentIndex][0] <= mutable.currentTime;
      mutable.currentIndex++
    ) {
      const message = recording.messages[mutable.currentIndex][1];
      viewerMutable.messageQueue.push(message);
    }

    // Don't loop for static scenes (durationSeconds === 0).
    if (
      mutable.currentTime >= recording.durationSeconds &&
      recording.durationSeconds > 0
    ) {
      mutable.currentIndex = 0;
      mutable.currentTime = recording.messages[0][0];
    }
    setCurrentTime(mutable.currentTime);
  }, [recording]);

  useEffect(() => {
    const playbackMultiplier = parseFloat(playbackSpeed);
    if (recording !== null && !paused) {
      let lastUpdate = Date.now();
      const interval = setInterval(() => {
        const now = Date.now();
        playbackMutable.current.currentTime +=
          ((now - lastUpdate) / 1000.0) * playbackMultiplier;
        lastUpdate = now;

        updatePlayback();
        // Stop playback for static scenes once all messages are processed.
        if (
          playbackMutable.current.currentIndex === recording.messages.length &&
          recording.durationSeconds === 0.0
        ) {
          clearInterval(interval);
        }
      }, 1000.0 / 120.0);
      return () => clearInterval(interval);
    }
  }, [
    updatePlayback,
    recording,
    paused,
    playbackSpeed,
    viewerMutable.messageQueue,
    setCurrentTime,
  ]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.code === "Space") {
        setPaused(!paused);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [paused]);

  const updateCurrentTime = useCallback(
    (value: number) => {
      if (value < playbackMutable.current.currentTime) {
        resetScene();
        playbackMutable.current.currentIndex = 0;
      }
      playbackMutable.current.currentTime = value;
      setCurrentTime(value);
      setPaused(true);
      updatePlayback();
    },
    [recording],
  );

  if (recording === null) {
    return (
      <div
        style={{
          position: "fixed",
          zIndex: 1,
          top: 0,
          bottom: 0,
          left: 0,
          right: 0,
          backgroundColor: darkMode ? theme.colors.dark[9] : "#fff",
        }}
      >
        <Progress
          value={(status.downloaded / status.total) * 100.0}
          radius={0}
          transitionDuration={0}
        />
      </div>
    );
  } else {
    return (
      <Paper
        radius="xs"
        shadow="0.1em 0 1em 0 rgba(0,0,0,0.1)"
        style={{
          position: "fixed",
          bottom: "1em",
          left: "50%",
          transform: "translateX(-50%)",
          width: "25em",
          maxWidth: "95%",
          zIndex: 1,
          padding: "0.5em",
          display: recording.durationSeconds === 0.0 ? "none" : "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.375em",
        }}
      >
        <ActionIcon
          size="md"
          variant="subtle"
          onClick={() => setPaused(!paused)}
        >
          {paused ? (
            <IconPlayerPlayFilled height="1.125em" width="1.125em" />
          ) : (
            <IconPlayerPauseFilled height="1.125em" width="1.125em" />
          )}
        </ActionIcon>
        <NumberInput
          size="xs"
          hideControls
          value={currentTime.toFixed(1)}
          step={0.01}
          styles={{
            wrapper: {
              width: "3.1em",
            },
            input: {
              padding: "0.2em",
              fontFamily: theme.fontFamilyMonospace,
              textAlign: "center",
            },
          }}
          onChange={(value) =>
            updateCurrentTime(
              typeof value === "number" ? value : parseFloat(value),
            )
          }
        />
        <Slider
          thumbSize={0}
          radius="xs"
          step={1e-4}
          style={{ flexGrow: 1 }}
          min={0}
          max={recording.durationSeconds}
          value={currentTime}
          onChange={updateCurrentTime}
          styles={{ thumb: { display: "none" } }}
        />
        <Tooltip zIndex={10} label={"Playback speed"} withinPortal>
          <Select
            size="xs"
            value={playbackSpeed}
            onChange={(val) => (val === null ? null : setPlaybackSpeed(val))}
            radius="xs"
            data={["0.5x", "1x", "2x", "4x", "8x"]}
            styles={{
              wrapper: { width: "3.25em" },
            }}
            comboboxProps={{ zIndex: 5, width: "5.25em" }}
          />
        </Tooltip>
      </Paper>
    );
  }
}
