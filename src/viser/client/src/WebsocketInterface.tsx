import AwaitLock from "await-lock";
import { unpack } from "msgpackr";

import React, { useContext } from "react";

import { ViewerContext } from "./App";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { Message } from "./WebsocketMessages";
/** Component for handling websocket connections. */
export function WebsocketMessageProducer() {
  const messageQueueRef = useContext(ViewerContext)!.messageQueueRef;
  const viewer = useContext(ViewerContext)!;
  const server = viewer.useGui((state) => state.server);
  const resetGui = viewer.useGui((state) => state.resetGui);

  syncSearchParamServer(server);

  React.useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let ws: null | WebSocket = null;
    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws = new WebSocket(server);

      // Timeout is necessary when we're connecting to an SSH/tunneled port.
      const retryTimeout = setTimeout(() => {
        ws?.close();
      }, 5000);

      ws.onopen = () => {
        clearTimeout(retryTimeout);
        console.log(`Connected! ${server}`);
        viewer.websocketRef.current = ws;
        viewer.useGui.setState({ websocketConnected: true });
      };

      ws.onclose = (event) => {
        console.log(`Disconnected! ${server} code=${event.code}`);
        clearTimeout(retryTimeout);
        viewer.websocketRef.current = null;
        viewer.scenePointerInfo.current!.enabled = false;
        viewer.useGui.setState({ websocketConnected: false });
        resetGui();

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      ws.onmessage = async (event) => {
        // Reduce websocket backpressure.
        const messagePromise = new Promise<Message[]>((resolve) => {
          (event.data.arrayBuffer() as Promise<ArrayBuffer>).then((buffer) => {
            resolve(unpack(new Uint8Array(buffer)) as Message[]);
          });
        });

        // Try our best to handle messages in order. If this takes more than 1 second, we give up. :)
        await orderLock.acquireAsync({ timeout: 1000 }).catch(() => {
          console.log("Order lock timed out.");
          orderLock.release();
        });
        try {
          const messages = await messagePromise;
          messageQueueRef.current.push(...messages);
        } finally {
          orderLock.acquired && orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      viewer.useGui.setState({ websocketConnected: false });
      ws?.close();
      clearTimeout(timeout);
    };
  }, [server, resetGui]);

  return <></>;
}
