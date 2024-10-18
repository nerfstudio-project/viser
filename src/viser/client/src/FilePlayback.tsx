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

  const baseSpeed = parseFloat(searchParams.get("baseSpeed") || "1");

  useEffect(() => {
    deserializeGzippedMsgpackFile<SerializedMessages>(fileUrl, setStatus).then(
      (record) => {
        // Apply baseSpeed.
        setRecording({
          loopStartIndex: record.loopStartIndex,
          durationSeconds: record.durationSeconds / baseSpeed,
          messages: record.messages.map(([time, message]) => [
            time / baseSpeed,
            message,
          ]),
        });
      },
    );
  }, []);

  const playbackMutable = useRef({
    currentTime: 0.0,
    currentIndex: 0,
    prevUpdateTime: 0.0,
  });

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

    if (mutable.currentTime < mutable.prevUpdateTime) {
      mutable.currentIndex = recording.loopStartIndex!;
    }
    mutable.prevUpdateTime = mutable.currentTime;
    setCurrentTime(mutable.currentTime);
  }, [recording]);

  useEffect(() => {
    const mutable = playbackMutable.current;
    const playbackMultiplier = parseFloat(playbackSpeed); // '0.5x' -> 0.5

    if (videoRef.current !== null) {
      if (paused && !videoRef.current.paused) {
        videoRef.current.pause();
      } else if (!paused && videoRef.current.paused) {
        videoRef.current.play();
      }
    }

    if (recording !== null && !paused) {
      let lastUpdate = Date.now();
      const interval = setInterval(() => {
        const now = Date.now();
        if (videoRef.current) {
          // Don't need to do any of this if there's a video.
          if (videoRef.current !== null && videoRef.current.readyState >= 2) {
            videoRef.current.playbackRate =
              baseSpeed * parseFloat(playbackSpeed);
            mutable.currentTime = videoRef.current.currentTime / baseSpeed;
            if (
              recording !== null &&
              mutable.currentTime > recording.durationSeconds
            ) {
              mutable.currentTime = 0;
              videoRef.current.currentTime = 0;
            }
            mutable.currentTime = Math.max(0, mutable.currentTime);
            updatePlayback();
          }
        } else {
          // Manually increment currentTime only if video is not available
          mutable.currentTime +=
            ((now - lastUpdate) / 1000.0) * playbackMultiplier;
          lastUpdate = now;
          if (mutable.currentTime > recording.durationSeconds) {
            mutable.currentTime = 0;
          }
          updatePlayback();
        }
        if (
          mutable.currentIndex === recording.messages.length &&
          recording.loopStartIndex === null
        ) {
          clearInterval(interval);
        }
      }, 1000.0 / 60.0);
      return () => clearInterval(interval);
    }
  }, [
    updatePlayback,
    recording,
    paused,
    playbackSpeed,
    messageQueueRef,
    setCurrentTime,
    baseSpeed,
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
      const mutable = playbackMutable.current;
      mutable.currentTime = value;
      setPaused(true);
      updatePlayback();

      // Update video time if video element exists
      if (videoRef.current) {
        videoRef.current.currentTime = value * baseSpeed;
      }
    },
    [recording, baseSpeed, updatePlayback],
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
              top: "0.75em",
              left: "0.75em",
              zIndex: 2,
              maxHeight: "auto",
              border: "0.2rem solid #fff",
              overflow: "hidden",
              borderRadius: "0.3rem",
              padding: "0",
            }}
          >
            <div
              style={{
                textAlign: "center",
                position: "absolute",
                top: "0",
                backgroundColor: "#fff",
                borderBottomRightRadius: "0.3rem",
                fontSize: "1.1em",
                color: "#000",
                fontFamily: "Inter",
                padding: "0.2em 0.5em",
                whiteSpace: "nowrap",
                fontWeight: "600",
              }}
            >
              Input video
            </div>
            <video
              ref={videoRef}
              src={overlayVideo[0] + `#t=${-videoTimeOffset}`}
              style={{
                width: "25rem",
                maxWidth: "23vw",
                minWidth: "10em",
                aspectRatio: "1",
                margin: "0",
                display: "block",
              }}
              loop
              autoPlay
              playsInline
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
            size="xxs"
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
                fontSize: "0.75em",
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
              size="xxs"
              value={playbackSpeed}
              onChange={(val) => (val === null ? null : setPlaybackSpeed(val))}
              radius="xs"
              data={["0.5x", "1x", "2x", "4x", "8x"]}
              styles={{
                wrapper: { width: "3.25em" },
                input: { fontSize: "0.75em", padding: "0.2em 0.75em" },
              }}
              comboboxProps={{ zIndex: 5, width: "5.25em" }}
            />
          </Tooltip>
        </Paper>
      </>
    );
  }
}
