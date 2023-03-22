from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from asyncio.events import AbstractEventLoop
from typing import Callable, Dict, List, NewType, Optional, Tuple

import websockets.connection
import websockets.exceptions
import websockets.server
from websockets.legacy.server import WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._message_api import MessageApi
from ._messages import Message, ViewerCameraMessage


@dataclasses.dataclass(frozen=True)
class CameraState:
    """Information about a client's camera state."""

    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    fov: float
    aspect: float
    last_updated: float


@dataclasses.dataclass
class _ClientHandleState:
    # Internal state for ClientHandle objects.
    camera_info: Optional[CameraState]
    message_buffer: asyncio.Queue
    event_loop: AbstractEventLoop
    camera_cb: List[Callable[[CameraState], None]]


@dataclasses.dataclass
class ClientHandle(MessageApi):
    """Handle for interacting with a single connected client.

    We can use this to read the camera state or send client-specific messages."""

    _state: _ClientHandleState

    def __post_init__(self) -> None:
        super().__init__()

        def handle_camera(client_id: ClientId, message: Message) -> None:
            """Handle camera messages."""
            if not isinstance(message, ViewerCameraMessage):
                return
            self._state.camera_info = CameraState(
                message.wxyz, message.position, message.fov, message.aspect, time.time()
            )
            for cb in self._state.camera_cb:
                cb(self._state.camera_info)

        self._incoming_handlers.append(handle_camera)

    def get_camera(self) -> CameraState:
        while self._state.camera_info is None:
            time.sleep(0.01)
        return self._state.camera_info

    def on_camera_update(
        self, callback: Callable[[CameraState], None]
    ) -> Callable[[CameraState], None]:
        self._state.camera_cb.append(callback)
        return callback

    def _queue(self, message: Message) -> None:
        """Implements message enqueue required by MessageApi.

        Pushes a message onto a client-specific queue."""
        self._state.event_loop.call_soon_threadsafe(
            self._state.message_buffer.put_nowait, message
        )


ClientId = NewType("ClientId", int)


class ViserServer(MessageApi):
    """Core visualization server. Communicates asynchronously with client applications
    via websocket connections.

    By default, all messages (eg `server.add_frame()`) are broadcasted to all connected
    clients.

    To send messages to an individual client, we can grab a client ID -> handle mapping
    via `server.get_clients()`, and then call `client.add_frame()` on the handle.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        super().__init__()

        # Track connected clients.
        self._handle_from_client: Dict[ClientId, _ClientHandleState] = {}
        self._client_lock = threading.Lock()

        # Start server thread.
        ready_sem = threading.Semaphore(value=1)
        ready_sem.acquire()
        threading.Thread(
            target=lambda: self._background_worker(host, port, ready_sem),
            daemon=True,
        ).start()

        # Wait for the thread to set self._event_loop and self._broadcast_buffer...
        ready_sem.acquire()

        # Broadcast buffer should be populated by the background worker.
        assert isinstance(self._broadcast_buffer, AsyncMessageBuffer)

        # Reset the scene.
        self.reset_scene()

    def get_clients(self) -> Dict[ClientId, ClientHandle]:
        """Get a mapping from client IDs to client handles.

        We can use client handles to get camera information, send individual messages to
        clients, etc."""

        self._client_lock.acquire()
        out = {k: ClientHandle(v) for k, v in self._handle_from_client.items()}
        self._client_lock.release()
        return out

    def _queue(self, message: Message) -> None:
        """Implements message enqueue required by MessageApi.

        Pushes a message onto a broadcast queue."""
        self._broadcast_buffer.push(message)

    def _background_worker(
        self, host: str, port: int, ready_sem: threading.Semaphore
    ) -> None:
        # Need to make a new event loop for notebook compatbility.
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        self._event_loop = event_loop
        self._broadcast_buffer = AsyncMessageBuffer(event_loop)
        ready_sem.release()

        count_lock = asyncio.Lock()
        connection_count = 0
        total_connections = 0

        async def serve(websocket: WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""

            async with count_lock:
                nonlocal connection_count
                client_id = ClientId(connection_count)
                connection_count += 1

                nonlocal total_connections
                total_connections += 1

            print(
                f"Connection opened ({client_id}, {total_connections} total),"
                f" {len(self._broadcast_buffer.message_from_id)} persistent messages"
            )

            client_handle = _ClientHandleState(
                camera_info=None,
                message_buffer=asyncio.Queue(),
                event_loop=event_loop,
                camera_cb=[],
            )
            self._client_lock.acquire()
            self._handle_from_client[client_id] = client_handle
            self._client_lock.release()

            def handle_incoming(message: Message) -> None:
                self._handle_incoming_message(client_id, message)
                ClientHandle(client_handle)._handle_incoming_message(client_id, message)

            try:
                # For each client: infinite loop over producers (which send messages)
                # and consumers (which receive messages).
                await asyncio.gather(
                    _single_connection_producer(
                        websocket, client_handle.message_buffer
                    ),
                    _broadcast_producer(websocket, client_id, self._broadcast_buffer),
                    _consumer(websocket, client_id, handle_incoming),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # Cleanup.
                print(f"Connection closed ({client_id}, {total_connections} total)")
                total_connections -= 1
                self._client_lock.acquire()
                self._handle_from_client.pop(client_id)
                self._client_lock.release()

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
    websocket: WebSocketServerProtocol, client_id: ClientId, buffer: AsyncMessageBuffer
) -> None:
    """Infinite loop to send messages from the broadcast buffer."""
    async for message in buffer:
        if message.excluded_self_client == client_id:
            continue
        await websocket.send(message.serialize())


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
