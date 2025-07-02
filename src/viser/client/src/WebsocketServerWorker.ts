import { encode, decode } from "@msgpack/msgpack";
import { Message } from "./WebsocketMessages";
import AwaitLock from "await-lock";
import { VISER_VERSION } from "./VersionInfo";

export type WsWorkerIncoming =
  | { type: "send"; message: Message }
  | { type: "set_server"; server: string }
  | { type: "close" };

export type WsWorkerOutgoing =
  | { type: "connected" }
  | {
      type: "closed";
      versionMismatch?: boolean;
      clientVersion?: string;
      closeReason?: string;
    }
  | { type: "message_batch"; messages: Message[] };

// Helper function to collect all ArrayBuffer objects. This is used for postMessage() move semantics.
function collectArrayBuffers(obj: any, buffers: Set<ArrayBufferLike>) {
  if (obj instanceof ArrayBuffer) {
    buffers.add(obj);
  } else if (obj instanceof Uint8Array) {
    buffers.add(obj.buffer);
  } else if (obj && typeof obj === "object") {
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        collectArrayBuffers(obj[key], buffers);
      }
    }
  }
  return buffers;
}

{
  let server: string | null = null;
  let ws: WebSocket | null = null;
  const orderLock = new AwaitLock();

  const postOutgoing = (
    data: WsWorkerOutgoing,
    transferable?: Transferable[],
  ) => {
    // @ts-ignore
    self.postMessage(data, transferable);
  };

  const tryConnect = () => {
    if (ws !== null) ws.close();

    // Use a single protocol that includes both client identification and version.
    const protocol = `viser-v${VISER_VERSION}`;
    console.log(`Connecting to: ${server!} with protocol: ${protocol}`);
    ws = new WebSocket(server!, [protocol]);

    // Timeout is necessary when we're connecting to an SSH/tunneled port.
    const retryTimeout = setTimeout(() => {
      ws?.close();
    }, 5000);

    ws.onopen = () => {
      clearTimeout(retryTimeout);
      console.log(`Connected! ${server}`);

      // Just indicate that we're connected.
      postOutgoing({
        type: "connected",
      });
    };

    ws.onclose = (event) => {
      // Check for explicit close (code 1002 = protocol error, which we use for version mismatch).
      const versionMismatch = event.code === 1002;

      // Send close notification.
      postOutgoing({
        type: "closed",
        versionMismatch: versionMismatch,
        clientVersion: VISER_VERSION,
        closeReason: event.reason || "Connection closed",
      });

      console.log(
        `Disconnected! ${server} code=${event.code}, reason: ${event.reason}`,
      );

      if (versionMismatch) {
        console.warn(
          `Connection rejected due to version mismatch. Client version: ${VISER_VERSION}`,
        );
      }

      clearTimeout(retryTimeout);

      // Try to reconnect on next repaint.
      // requestAnimationFrame() helps us avoid reconnecting from tabs that are
      // hidden or minimized.
      if (server !== null) {
        requestAnimationFrame(() => {
          setTimeout(tryConnect, 1000);
        });
      }
    };

    // Track ideal send time for message smoothing.
    // State for tracking message timing and smoothing.
    const state: {
      prevPythonTimestampMs?: number;
      lastIdealSendTimeMs?: number;
      firstPythonTimestampMs?: number;
      firstJsTimeMs?: number;
    } = {};
    type SerializedStruct = {
      messages: Message[];
      timestampSec: number;
    };

    ws.onmessage = async (event) => {
      const dataPromise = new Promise<SerializedStruct>((resolve) => {
        (event.data.arrayBuffer() as Promise<ArrayBuffer>).then((buffer) => {
          resolve(decode(new Uint8Array(buffer)) as SerializedStruct);
        });
      });

      // Try our best to handle messages in order. If this takes more than 10 seconds, we give up. :)
      await orderLock.acquireAsync({ timeout: 10000 }).catch(() => {
        console.log("Order lock timed out.");
        orderLock.release();
      });
      const data = await dataPromise;

      // Function to send the message and release the order lock.
      const messages = data.messages;
      const arrayBuffers = collectArrayBuffers(messages, new Set());
      const sendFn = () => {
        // Update the state with the latest timestamps.
        state.prevPythonTimestampMs = currentPythonTimestampMs;
        postOutgoing(
          { type: "message_batch", messages: messages },
          Array.from(arrayBuffers),
        );
        orderLock.release();
      };

      // Calculate timing deltas between Python and JavaScript.
      const jsReceiveTimeMs = performance.now();
      const currentPythonTimestampMs = data.timestampSec * 1000;
      const pythonTimeDeltaMs =
        currentPythonTimestampMs -
        (state.prevPythonTimestampMs ?? currentPythonTimestampMs);

      // Establish reference point on first message.
      if (state.firstPythonTimestampMs === undefined) {
        state.firstPythonTimestampMs = currentPythonTimestampMs;
        state.firstJsTimeMs = jsReceiveTimeMs;
      }

      // Calculate how far behind real-time we are.
      const pythonElapsedMs =
        currentPythonTimestampMs - state.firstPythonTimestampMs;
      const jsElapsedMs = jsReceiveTimeMs - state.firstJsTimeMs!;
      const accumulatedDelayMs = jsElapsedMs - pythonElapsedMs;

      if (
        // Flush immediately for first message.
        state.lastIdealSendTimeMs === undefined ||
        state.prevPythonTimestampMs === undefined ||
        // Flush immediately if the Python delta is large, in this case we're
        // rarely sensitive to exact timing.
        currentPythonTimestampMs - state.prevPythonTimestampMs > 100 ||
        // Flush if we're more than 200ms behind real-time.
        accumulatedDelayMs > 200
      ) {
        // First message or no expected delta, send immediately.
        sendFn();
        state.lastIdealSendTimeMs = jsReceiveTimeMs;
      } else {
        // For messages that are being sent frequently: smooth out the sending rate.
        const idealNextSendTimeMs =
          state.lastIdealSendTimeMs + pythonTimeDeltaMs;
        const timeUntilExpectedMs = Math.min(
          idealNextSendTimeMs - jsReceiveTimeMs,
          100,
        );

        const earlyThresholdMs = 5;
        if (timeUntilExpectedMs > earlyThresholdMs) {
          // We're early! This means the previous message arrived late...
          // we can send this one later to make the client time delta
          // more consistent with the Python time delta.
          console.log(timeUntilExpectedMs);
          setTimeout(sendFn, timeUntilExpectedMs - earlyThresholdMs);
          state.lastIdealSendTimeMs =
            jsReceiveTimeMs + timeUntilExpectedMs - earlyThresholdMs;
        } else {
          // Message is on time or late: send immediately.
          sendFn();
          state.lastIdealSendTimeMs =
            state.lastIdealSendTimeMs + pythonTimeDeltaMs;
        }
      }
    };
  };

  self.onmessage = (e) => {
    const data: WsWorkerIncoming = e.data;

    if (data.type === "send") {
      ws!.send(encode(data.message));
    } else if (data.type === "set_server") {
      server = data.server;
      tryConnect();
    } else if (data.type == "close") {
      server = null;
      ws !== null && ws.close();
      self.close();
    } else {
      console.log(
        `WebSocket worker: got ${data}, not sure what to do with it!`,
      );
    }
  };
}
