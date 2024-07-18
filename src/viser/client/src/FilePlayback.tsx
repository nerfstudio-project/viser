import { decodeAsync, decode } from "@msgpack/msgpack";
import { Message } from "./WebsocketMessages";
import { decompress } from "fflate";

import { useContext, useEffect, useState } from "react";
import { ViewerContext } from "./App";
import { Progress, useMantineTheme } from "@mantine/core";

interface SerializedMessages {
  loopStartIndex: number | null;
  durationSeconds: number;
  messages: [number, Message][];
}

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

export function PlaybackFromFile({ fileUrl }: { fileUrl: string }) {
  const viewer = useContext(ViewerContext)!;
  const messageQueueRef = viewer.messageQueueRef;

  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const [status, setStatus] = useState({ downloaded: 0.0, total: 0.0 });
  const [loaded, setLoaded] = useState(false);
  const theme = useMantineTheme();

  useEffect(() => {
    deserializeGzippedMsgpackFile<SerializedMessages>(fileUrl, setStatus).then(
      (recording) => {
        let messageIndex = 0;

        function continuePlayback() {
          setLoaded(true);
          const currentTimeSeconds = recording.messages[messageIndex][0];
          while (currentTimeSeconds >= recording.messages[messageIndex][0]) {
            messageQueueRef.current.push(recording.messages[messageIndex][1]);
            messageIndex += 1;

            // Either finish playback or loop.
            if (messageIndex === recording.messages.length) {
              if (recording.loopStartIndex === null) return;
              messageIndex = recording.loopStartIndex;
              setTimeout(
                continuePlayback,
                (recording.durationSeconds - currentTimeSeconds) * 1000.0,
              );
              return;
            }
          }

          // Handle next set of frames.
          setTimeout(
            continuePlayback,
            (recording.messages[messageIndex][0] - currentTimeSeconds) * 1000.0,
          );
        }
        setTimeout(continuePlayback, recording.messages[0][0] * 1000.0);
      },
    );
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        zIndex: 1000,
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
        display: loaded ? "none" : "block",
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
}
