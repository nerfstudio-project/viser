import React from "react";
import * as THREE from "three";
import { Message } from "./WebsocketMessages";
import { ViewerContext, ViewerContextContents } from "./ViewerContext";

/** Easier, hook version of makeThrottledMessageSender. */
export function useThrottledMessageSender(throttleMilliseconds: number) {
  const viewer = React.useContext(ViewerContext)!;
  return makeThrottledMessageSender(viewer, throttleMilliseconds);
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  viewer: ViewerContextContents,
  throttleMilliseconds: number,
) {
  let readyToSend = true;
  let stale = false;
  let latestMessage: Message | null = null;

  function send(message: Message) {
    const viewerMutable = viewer.mutable.current;
    if (viewerMutable.sendMessage === null) return;
    latestMessage = message;
    if (readyToSend) {
      viewerMutable.sendMessage(message);
      stale = false;
      readyToSend = false;

      setTimeout(() => {
        readyToSend = true;
        if (!stale) return;
        latestMessage && send(latestMessage);
      }, throttleMilliseconds);
    } else {
      stale = true;
    }
  }
  function flush() {
    const viewerMutable = viewer.mutable.current;
    if (viewerMutable.sendMessage === null) return;
    if (latestMessage !== null) {
      viewer.mutable.current.sendMessage(latestMessage);
      latestMessage = null;
      stale = false;
    }
  }
  return { send, flush };
}

/** Type guard for threejs textures. Meant to be used with `scene.background`. */
export function isTexture(
  background:
    | THREE.Color
    | THREE.Texture
    | THREE.CubeTexture
    | null
    | undefined,
): background is THREE.Texture {
  return (
    background !== null &&
    background !== undefined &&
    (background as THREE.Texture).isTexture !== undefined
  );
}
