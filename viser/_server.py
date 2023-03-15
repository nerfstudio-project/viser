#!/usr/bin/env python


import asyncio
import multiprocessing
import queue
import threading
import time
from typing import Optional

import msgpack
import websockets.connection
import websockets.exceptions
import websockets.server
from websockets.legacy.server import WebSocketServer, WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._messages import Message, ViewerCameraMessage


class ViserServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        # Create websocket server process.
        self.message_queue = queue.Queue(maxsize=1024)
        self.camera: Optional[ViewerCameraMessage] = None
        self.camera_timestamp: float = 0.0
        threading.Thread(
            target=self._start_background_loop,
            args=(host, port, self.message_queue),
            daemon=True,
        ).start()

    def queue(self, *messages: Message) -> None:
        """Queue a message, which will be sent to all clients."""
        for message in messages:
            self.message_queue.put_nowait(message)

    def _start_background_loop(
        self,
        host: str,
        port: int,
        message_queue: multiprocessing.Queue,
    ) -> None:
        connection_count = 0
        total_connections = 0
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        message_buffer = AsyncMessageBuffer(event_loop)

        def message_transfer() -> None:
            """Message transfer loop. Pulls messages from the main process and pushes them into our buffer."""
            while True:
                message_buffer.push(message_queue.get())

        async def serve(websocket: WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""

            # TODO: there are likely race conditions here...
            nonlocal connection_count
            connection_id = connection_count
            connection_count += 1

            nonlocal total_connections
            total_connections += 1

            print(
                f"Connection opened ({connection_id}, {total_connections} total),"
                f" {len(message_buffer.message_from_id)} buffered messages"
            )
            try:
                await asyncio.gather(
                    producer(websocket),
                    consumer(websocket),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                total_connections -= 1
                print(f"Connection closed ({connection_id}, {total_connections} total)")

        async def producer(websocket: WebSocketServerProtocol) -> None:
            # Infinite loop to send messages from the message buffer.
            async for message in message_buffer:
                await websocket.send(message)

        async def consumer(websocket: WebSocketServerProtocol) -> None:
            while True:
                message = msgpack.unpackb(await websocket.recv())
                t = message.pop("type")
                if t == "viewer_camera":
                    self.camera = ViewerCameraMessage(**message)
                    self.camera_timestamp = time.time()
                else:
                    print("Unrecognized message", message)

        # Start message transfer thread.
        threading.Thread(target=message_transfer).start()

        # Run server.
        event_loop.run_until_complete(websockets.server.serve(serve, host, port))
        event_loop.run_forever()
