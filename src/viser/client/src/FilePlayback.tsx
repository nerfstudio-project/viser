import { decodeAsync, decode } from "@msgpack/msgpack";
import { Message } from "./WebsocketMessages";
import { decompress } from "fflate";

import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { ViewerContext } from "./App";
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

interface SerializedMessages {
  loopStartIndex: number | null;
  durationSeconds: number;
  messages: [number, Message][];
}

export function PlaybackFromFile({ fileUrl }: { fileUrl: string }) {
  const viewer = useContext(ViewerContext)!;
  const messageQueueRef = viewer.messageQueueRef;

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const [status, setStatus] = useState({ downloaded: 0.0, total: 0.0 });
  const [playbackSpeed, setPlaybackSpeed] = useState("1x");
  const [paused, setPaused] = useState(false);
  const [recording, setRecording] = useState<SerializedMessages | null>(null);

  const searchParams = new URLSearchParams(window.location.search);
  const overlayVideo = searchParams.getAll("synchronizedVideoOverlay");
  const videoTimeOffset = parseFloat(
    searchParams.get("synchronizedVideoTimeOffset") || "0",
  );

  const [currentTime, setCurrentTime] = useState(0.0);

  const theme = useMantineTheme();

  useEffect(() => {
    deserializeGzippedMsgpackFile<SerializedMessages>(fileUrl, setStatus).then(
      setRecording,
    );
  }, []);

  const playbackMutable = useRef({ currentTime: 0.0, currentIndex: 0 });

  const updatePlayback = useCallback(() => {
    if (recording === null) return;
    const mutable = playbackMutable.current;

    // We have messages with times: [0.0, 0.01, 0.01, 0.02, 0.03]
    // We have our current time: 0.02
    // We want to get of a slice of all message _until_ the current time.
    for (
      ;
      mutable.currentIndex < recording.messages.length &&
      recording.messages[mutable.currentIndex][0] <= mutable.currentTime;
      mutable.currentIndex++
    ) {
      const message = recording.messages[mutable.currentIndex][1];
      messageQueueRef.current.push(message);
    }

    if (
      mutable.currentTime >= recording.durationSeconds &&
      recording.loopStartIndex !== null
    ) {
      mutable.currentIndex = recording.loopStartIndex!;
      mutable.currentTime = recording.messages[recording.loopStartIndex!][0];
    }
    setCurrentTime(mutable.currentTime);
  }, [recording]);

  useEffect(() => {
    let animationFrameId: number;

    const updateVideoTime = () => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        videoRef.current.currentTime = Math.max(
          0,
          playbackMutable.current.currentTime + videoTimeOffset,
        );
      }
      animationFrameId = requestAnimationFrame(updateVideoTime);
    };
    updateVideoTime();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, []);

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
        if (
          playbackMutable.current.currentIndex === recording.messages.length &&
          recording.loopStartIndex === null
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
    messageQueueRef,
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
        playbackMutable.current.currentIndex = recording!.loopStartIndex!;
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
      <>
        {overlayVideo.length > 0 && (
          <div
            style={{
              position: "fixed",
              top: "1.5em",
              left: "1em",
              zIndex: 2,
              width: "15rem",
              height: "15rem",
              boxShadow: "0 0 0.5em rgba(255, 255, 255, 0.5)",
              overflow: "hidden",
              borderRadius: "0.25em",
            }}
          >
            <div
              style={{
                textAlign: "center",
                position: "absolute",
                top: "0",
                width: "100%",
                backgroundColor: "rgba(0,0,0,0.5)",
                lineHeight: "2em",
                fontFamily: "Inter",
                padding: "0.5em 1em 0 1em",
                fontWeight: "500",
              }}
            >
              Input video
            </div>
            <video
              ref={videoRef}
              src={overlayVideo[0]}
              style={{
                width: "100%",
                height: "100%",
              }}
              muted
            />
          </div>
        )}
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
          <ActionIcon size="md" variant="subtle">
            {paused ? (
              <IconPlayerPlayFilled
                onClick={() => setPaused(false)}
                height="1.125em"
                width="1.125em"
              />
            ) : (
              <IconPlayerPauseFilled
                onClick={() => setPaused(true)}
                height="1.125em"
                width="1.125em"
              />
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
      </>
    );
  }
}
