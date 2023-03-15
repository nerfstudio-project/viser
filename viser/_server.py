#!/usr/bin/env python

import asyncio
import dataclasses
import queue as queue_
import threading
import time
from typing import Dict, List, Literal, Optional, Union

import msgpack
import websockets.connection
import websockets.exceptions
import websockets.server
from websockets.legacy.server import WebSocketServer, WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._messages import Message, ViewerCameraMessage


@dataclasses.dataclass
class Connection:
    camera: Optional[ViewerCameraMessage]
    camera_timestamp: float
    _message_buffer: asyncio.Queue


class ViserServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        # Create websocket server process.
        self.connection_from_id: Dict[int, Connection] = {}
        self._message_queue = queue_.Queue(maxsize=1024)
        threading.Thread(
            target=self._start_background_loop,
            args=(host, port, self._message_queue),
            daemon=True,
        ).start()

    def queue(
        self,
        *messages: Message,
        connection_id: Union[Literal["broadcast"], int] = "broadcast",
    ) -> None:
        """Queue a message, which can either be broadcast or sent to a particular connection ID."""
        for message in messages:
            self._message_queue.put_nowait((connection_id, message))

    def _start_background_loop(
        self,
        host: str,
        port: int,
        message_queue: queue_.Queue,
    ) -> None:
        connection_count = 0
        total_connections = 0
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        # We maintain two message buffer types: a persistent broadcasted messages and
        # connection-specific messages.
        broadcast_buffer = AsyncMessageBuffer(event_loop)

        def message_transfer() -> None:
            """Message transfer loop. Pulls messages from the main process and pushes them into our buffer."""
            while True:
                connection_id, message = message_queue.get()
                if connection_id == "broadcast":
                    broadcast_buffer.push(message)
                elif connection_id in self.connection_from_id:
                    event_loop.call_soon_threadsafe(
                        self.connection_from_id[
                            connection_id
                        ]._message_buffer.put_nowait,
                        message,
                    )
                else:
                    print(
                        f"Tried to send message to {connection_id}, but ID is closed or not valid."
                    )

        async def serve(websocket: WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""

            # TODO: there are likely race conditions here...
            nonlocal connection_count
            connection_id = connection_count
            connection_count += 1

            nonlocal total_connections
            total_connections += 1

            single_connection_buffer = asyncio.Queue()
            connection = Connection(None, 0.0, single_connection_buffer)
            self.connection_from_id[connection_id] = connection

            print(
                f"Connection opened ({connection_id}, {total_connections} total),"
                f" {len(broadcast_buffer.message_from_id)} buffered messages"
            )
            try:
                await asyncio.gather(
                    single_connection_producer(websocket, single_connection_buffer),
                    broadcast_producer(websocket),
                    consumer(websocket, connection),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                total_connections -= 1
                print(f"Connection closed ({connection_id}, {total_connections} total)")
                self.connection_from_id.pop(connection_id)

        async def single_connection_producer(
            websocket: WebSocketServerProtocol, buffer: asyncio.Queue
        ) -> None:
            # Infinite loop to send messages from the message buffer.
            while True:
                message = await buffer.get()
                await websocket.send(message.serialize())

        async def broadcast_producer(websocket: WebSocketServerProtocol) -> None:
            # Infinite loop to send messages from the message buffer.
            async for message_serialized in broadcast_buffer:
                await websocket.send(message_serialized)

        async def consumer(
            websocket: WebSocketServerProtocol, connection: Connection
        ) -> None:
            while True:
                message = msgpack.unpackb(await websocket.recv())
                t = message.pop("type")
                if t == "viewer_camera":
                    connection.camera = ViewerCameraMessage(**message)
                    connection.camera_timestamp = time.time()
                else:
                    print("Unrecognized message", message)

        # Start message transfer thread.
        threading.Thread(target=message_transfer).start()

        # Run server.
        event_loop.run_until_complete(websockets.server.serve(serve, host, port))
        event_loop.run_forever()
