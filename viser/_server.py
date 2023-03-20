from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from typing import Callable, Dict, Literal, NewType, Optional, Tuple, Union

import websockets.connection
import websockets.exceptions
import websockets.server
from websockets.legacy.server import WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._messages import Message, ViewerCameraMessage


@dataclasses.dataclass(frozen=True)
class ClientInfo:
    """Information attached to a websocket connection."""

    camera: ViewerCameraMessage
    camera_timestamp: float


ClientId = NewType("ClientId", int)


class ViserServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        # Note that we maintain two message buffer types: a persistent broadcasted messages and
        # connection-specific messages.
        self._buffer_from_client: Dict[ClientId, asyncio.Queue] = {}
        self._info_from_client: Dict[ClientId, ClientInfo] = {}

        # Start server thread.
        ready_sem = threading.Semaphore(value=1)
        ready_sem.acquire()
        threading.Thread(
            target=lambda: self._background_worker(host, port, ready_sem),
            daemon=True,
        ).start()

        # Wait for the thread to set self._event_loop and self._broadcast_buffer...
        ready_sem.acquire()

    def get_client_ids(self) -> Tuple[ClientId, ...]:
        """Get a tuple of all connected client IDs."""
        return tuple(self._info_from_client.keys())

    def get_client_info(self, client_id: ClientId) -> Optional[ClientInfo]:
        """Get information (currently just camera pose) associated with a particular client."""
        return self._info_from_client.get(client_id, None)

    def queue(
        self,
        *messages: Message,
        client_id: Union[Literal["broadcast"], ClientId] = "broadcast",
    ) -> None:
        """Queue a message, which can either be broadcast or sent to a particular connection ID.

        Broadcasted messages are persistent and will be received by new connections."""
        if client_id == "broadcast":
            for m in messages:
                self._broadcast_buffer.push(m)
        else:
            client_buffer = self._buffer_from_client.get(client_id, None)
            if client_buffer is None:
                print(f"Client {client_id} is not connected! Ignoring message!")
                return
            for m in messages:
                self._event_loop.call_soon_threadsafe(client_buffer.put_nowait, m)

    def _handle_incoming_message(self, client_id: ClientId, message: Message) -> None:
        """Handler for incoming messages."""
        if isinstance(message, ViewerCameraMessage):
            self._info_from_client[client_id] = ClientInfo(
                camera=message,
                camera_timestamp=time.time(),
            )
        else:
            print("Unrecognized message", message)

    def _background_worker(
        self, host: str, port: int, ready_sem: threading.Semaphore
    ) -> None:
        # Need to make a new event loop for notebook compatbility.
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        self._event_loop = event_loop
        self._broadcast_buffer = AsyncMessageBuffer(event_loop)
        ready_sem.release()

        connection_count = 0
        total_connections = 0

        async def serve(websocket: WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""

            # TODO: there are likely race conditions here...
            nonlocal connection_count
            client_id = ClientId(connection_count)
            connection_count += 1

            nonlocal total_connections
            total_connections += 1

            self._buffer_from_client[client_id] = asyncio.Queue()

            print(
                f"Connection opened ({client_id}, {total_connections} total),"
                f" {len(self._broadcast_buffer.message_from_id)} persistent messages"
            )
            try:
                # For each client: infinite loop over producers (which send messages)
                # and consumers (which receive messages).
                await asyncio.gather(
                    _single_connection_producer(
                        websocket, self._buffer_from_client[client_id]
                    ),
                    _broadcast_producer(websocket, self._broadcast_buffer),
                    _consumer(
                        websocket,
                        client_id,
                        lambda message: self._handle_incoming_message(
                            client_id, message
                        ),
                    ),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # Cleanup.
                print(f"Connection closed ({client_id}, {total_connections} total)")
                total_connections -= 1
                self._buffer_from_client.pop(client_id)
                if client_id in self._info_from_client:
                    self._info_from_client.pop(client_id)

        # Run server.
        event_loop.run_until_complete(websockets.server.serve(serve, host, port))
        event_loop.run_forever()


async def _single_connection_producer(
    websocket: WebSocketServerProtocol, buffer: asyncio.Queue
) -> None:
    """Infinite loop to send messages from the client buffer."""
    while True:
        message = await buffer.get()
        await websocket.send(message.serialize())


async def _broadcast_producer(
    websocket: WebSocketServerProtocol, buffer: AsyncMessageBuffer
) -> None:
    """Infinite loop to send messages from the broadcast buffer."""
    async for message_serialized in buffer:
        await websocket.send(message_serialized)


async def _consumer(
    websocket: WebSocketServerProtocol,
    client_id: ClientId,
    handle_message: Callable[[Message], None],
) -> None:
    """Infinite loop waiting for and then handling incoming messages."""
    while True:
        raw = await websocket.recv()
        assert isinstance(raw, bytes)
        message = Message.deserialize(raw)
        handle_message(message)
