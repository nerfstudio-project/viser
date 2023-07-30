import { useFrame } from "@react-three/fiber";
import { unpack } from "msgpackr";
import { Message } from "./WebsocketMessages";
import React from "react";
import { useMessageHandler } from "./MessageHandler";
import * as pako from "pako";

interface SerializedMessages {
  loopStartIndex: number | null;
  durationSeconds: number;
  messages: [number, Message][];
}

async function deserializeMsgpackFile<T>(fileUrl: string): Promise<T> {
  const response = await fetch(fileUrl, {});
  const buffer = await response.arrayBuffer();

  // There is likely some parallelization we could do here.
  let uncompressedBuffer: Uint8Array;
  if (fileUrl.endsWith(".gz")) {
    const compressedBuffer = new Uint8Array(buffer);
    uncompressedBuffer = pako.inflate(compressedBuffer);
  } else {
    uncompressedBuffer = new Uint8Array(buffer);
  }

  return unpack(uncompressedBuffer);
}

export function PlaybackFromFile({ fileUrl }: { fileUrl: string }) {
  const handleMessage = useMessageHandler();

  const state = React.useRef<{
    loaded: SerializedMessages;
    index: number;
    startTimeSeconds: number;
  } | null>(null);

  deserializeMsgpackFile<SerializedMessages>(fileUrl).then((loaded) => {
    state.current = {
      loaded: loaded,
      index: 0,
      startTimeSeconds: Date.now() / 1000.0,
    };
  });

  useFrame(() => {
    const currentState = state.current;
    if (currentState === null) return;

    // Get seconds elapsed since start. We offset by the first message's
    // timestamp.
    const elapsedSeconds = Date.now() / 1000.0 - currentState.startTimeSeconds;

    // Handle messages.
    while (
      currentState.index < currentState.loaded.messages.length &&
      currentState.loaded.messages[currentState.index][0] <= elapsedSeconds
    ) {
      const msg = currentState.loaded.messages[currentState.index][1];
      handleMessage(msg);
      currentState.index += 1;
    }

    // Reset if looping.
    if (
      currentState.loaded.loopStartIndex !== null &&
      elapsedSeconds >= currentState.loaded.durationSeconds
    ) {
      currentState.index = currentState.loaded.loopStartIndex;
      currentState.startTimeSeconds =
        Date.now() / 1000.0 -
        currentState.loaded.messages[currentState.index][0];
      return;
    }
  });

  return <></>;
}
