import WebsocketServerWorker from "./WebsocketServerWorker?worker";
import React, { useContext } from "react";

import { ViewerContext } from "./App";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { WsWorkerIncoming, WsWorkerOutgoing } from "./WebsocketServerWorker";

/** Component for handling websocket connections. */
export function WebsocketMessageProducer() {
  const messageQueueRef = useContext(ViewerContext)!.messageQueueRef;
  const viewer = useContext(ViewerContext)!;
  const server = viewer.useGui((state) => state.server);
  const resetGui = viewer.useGui((state) => state.resetGui);
  const resetScene = viewer.useSceneTree((state) => state.resetScene);

  syncSearchParamServer(server);

  React.useEffect(() => {
    const worker = new WebsocketServerWorker();

    worker.onmessage = (event) => {
      const data: WsWorkerOutgoing = event.data;
      if (data.type === "connected") {
        resetGui();
        resetScene();
        viewer.useGui.setState({ websocketConnected: true });
        viewer.sendMessageRef.current = (message) => {
          postToWorker({ type: "send", message: message });
        };
      } else if (data.type === "closed") {
        resetGui();
        viewer.useGui.setState({ websocketConnected: false });
        viewer.sendMessageRef.current = (message) => {
          console.log(
            `Tried to send ${message.type} but websocket is not connected!`,
          );
        };
      } else if (data.type === "message_batch") {
        messageQueueRef.current.push(...data.messages);
      }
    };
    function postToWorker(data: WsWorkerIncoming) {
      worker.postMessage(data);
    }
    postToWorker({ type: "set_server", server: server });
    return () => {
      postToWorker({ type: "close" });
      viewer.sendMessageRef.current = (message) =>
        console.log(
          `Tried to send ${message.type} but websocket is not connected!`,
        );
      viewer.useGui.setState({ websocketConnected: false });
    };
  }, [server, resetGui, resetScene]);

  return <></>;
}
