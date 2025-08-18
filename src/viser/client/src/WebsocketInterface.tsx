import WebsocketServerWorker from "./WebsocketServerWorker?worker";
import React, { useContext } from "react";
import { notifications } from "@mantine/notifications";

import { ViewerContext } from "./ViewerContext";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { WsWorkerIncoming, WsWorkerOutgoing } from "./WebsocketServerWorker";

/** Component for handling websocket connections. */
export function WebsocketMessageProducer() {
  const viewer = useContext(ViewerContext)!;
  const viewerMutable = viewer.mutable.current;
  const server = viewer.useGui((state) => state.server);
  const resetGui = viewer.useGui((state) => state.resetGui);
  const resetScene = viewer.sceneTreeActions.resetScene;

  syncSearchParamServer(server);

  React.useEffect(() => {
    const worker = new WebsocketServerWorker();

    worker.onmessage = (event) => {
      const data: WsWorkerOutgoing = event.data;
      if (data.type === "connected") {
        resetGui();
        resetScene();
        viewer.useGui.setState({ websocketConnected: true });
        viewerMutable.sendMessage = (message) => {
          postToWorker({ type: "send", message: message });
        };
      } else if (data.type === "closed") {
        resetGui();
        viewer.useGui.setState({ websocketConnected: false });
        viewerMutable.sendMessage = (message) => {
          console.log(
            `Tried to send ${message.type} but websocket is not connected!`,
          );
        };

        // Show notification for version mismatch.
        if (data.versionMismatch) {
          notifications.show({
            id: "version-mismatch",
            title: "Connection rejected",
            message: `${data.closeReason}.`,
            color: "red",
            autoClose: 5000,
            withCloseButton: true,
          });
        }
      } else if (data.type === "message_batch") {
        viewerMutable.messageQueue.push(...data.messages);
      }
    };
    function postToWorker(data: WsWorkerIncoming) {
      worker.postMessage(data);
    }
    postToWorker({ type: "set_server", server: server });
    return () => {
      postToWorker({ type: "close" });
      viewerMutable.sendMessage = (message) =>
        console.log(
          `Tried to send ${message.type} but websocket is not connected!`,
        );
      viewer.useGui.setState({ websocketConnected: false });
    };
  }, [server, resetGui, resetScene]);

  return null;
}
