import { MutableRefObject } from "react";
import { Message } from "./WebsocketMessages";
import { pack } from "msgpackr";
import * as React from "react";
import { Progress } from "@mantine/core";
import { IconCheck } from "@tabler/icons-react";
import { v4 as uuid } from "uuid";
import { notifications } from "@mantine/notifications";

/** Send message over websocket. */
export function sendWebsocketMessage(
  websocketRef: MutableRefObject<WebSocket | null>,
  message: Message,
) {
  if (websocketRef.current === null) return;
  websocketRef.current.send(pack(message));
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  websocketRef: MutableRefObject<WebSocket | null>,
  throttleMilliseconds: number,
) {
  let readyToSend = true;
  let stale = false;
  let latestMessage: Message | null = null;

  function send(message: Message) {
    if (websocketRef.current === null) return;
    latestMessage = message;
    if (readyToSend) {
      websocketRef.current.send(pack(message));
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
  return send;
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

export function useFileUpload({
  websocketRef,
}: {
  websocketRef: MutableRefObject<WebSocket | null>,
}) {
  const [isUploading, setIsUploading] = React.useState<boolean>(false);

  async function upload({ file, componentId }: { file: File, componentId: string }) {
    const chunkSize = 512 * 1024; // bytes
    const numChunks = Math.ceil(file.size / chunkSize);
    const transferUuid = `${componentId}/${uuid()}`;
    const notificationId = "upload-" + transferUuid;
    const send = (message: Message) => websocketRef.current?.send(pack(message));

    let displaySize = file.size;
    const displayUnits = ["B", "K", "M", "G", "T", "P"];
    let displayUnitIndex = 0;
    while (
      displaySize >= 100 &&
      displayUnitIndex < displayUnits.length - 1
    ) {
      displaySize /= 1024;
      displayUnitIndex += 1;
    }
    const displaySizeString = `${displaySize.toFixed(1)}${displayUnits[displayUnitIndex]}`

    // Show notification.
    notifications.show({
      id: notificationId,
      title: "Uploading " + `${file.name} (${displaySizeString})`,
      message: <Progress size="sm" value={0} />,
      autoClose: false,
      withCloseButton: false,
      loading: true,
    });

    // Set uploading state
    setIsUploading(true);
    send({
      type: "FileTransferStart",
      transfer_uuid: transferUuid,
      filename: file.name,
      size_bytes: file.size,
      mime_type: file.type,
      part_count: numChunks,
    });

    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      const end = (i + 1) * chunkSize;
      const chunk = file.slice(start, end);
      const buffer = await chunk.arrayBuffer();
      const progressValue = (1 + i) / numChunks;

      send({
        type: "FileTransferPart",
        transfer_uuid: transferUuid,
        part: i,
        content: new Uint8Array(buffer),
      });

      notifications.update({
        id: notificationId,
        title: "Uploading " + `${file.name} (${displaySizeString})`,
        message: <Progress size="sm" value={100 * progressValue} />,
        autoClose: false,
        withCloseButton: false,
        loading: true,
      });
    }

    // Upload finished
    setIsUploading(false);
    notifications.update({
      id: notificationId,
      title: "Uploaded " + `${file.name} (${displaySizeString})`,
      message: "File uploaded successfully.",
      autoClose: true,
      withCloseButton: true,
      loading: false,
      icon: <IconCheck />,
    });
  }

  return {
    isUploading,
    upload
  }
}
