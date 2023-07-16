from __future__ import annotations

import asyncio
import dataclasses
import http.server
import mimetypes
import threading
from asyncio.events import AbstractEventLoop
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Type, TypeVar

import rich
import websockets.connection
import websockets.datastructures
import websockets.exceptions
import websockets.server
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from websockets.legacy.server import WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._messages import Message


@dataclasses.dataclass
class _ClientHandleState:
    # Internal state for ClientConnection objects.
    message_buffer: asyncio.Queue
    event_loop: AbstractEventLoop


ClientId = NewType("ClientId", int)
TMessage = TypeVar("TMessage", bound=Message)


class MessageHandler:
    """Mix-in for adding message handling to a class."""

    def __init__(self) -> None:
        self._incoming_handlers: Dict[
            Type[Message], List[Callable[[ClientId, Message], None]]
        ] = {}

    def register_handler(
        self,
        message_cls: Type[TMessage],
        callback: Callable[[ClientId, TMessage], None],
    ) -> None:
        """Register a handler for a particular message type."""
        if message_cls not in self._incoming_handlers:
            self._incoming_handlers[message_cls] = []
        self._incoming_handlers[message_cls].append(callback)  # type: ignore

    def _handle_incoming_message(self, client_id: ClientId, message: Message) -> None:
        """Handle incoming messages."""
        if type(message) in self._incoming_handlers:
            for cb in self._incoming_handlers[type(message)]:
                cb(client_id, message)


@dataclasses.dataclass
class ClientConnection(MessageHandler):
    """Handle for interacting with a single connected client.

    We can use this to read the camera state or send client-specific messages."""

    client_id: ClientId
    _state: _ClientHandleState

    def __post_init__(self) -> None:
        super().__init__()

    def send(self, message: Message) -> None:
        """Send a message to a specific client."""
        self._state.event_loop.call_soon_threadsafe(
            self._state.message_buffer.put_nowait, message
        )


class Server(MessageHandler):
    """Websocket server abstraction. Communicates asynchronously with client
    applications.

    By default, all messages are broadcasted to all connected clients.

    To send messages to an individual client, we can use `on_client_connect()` to
    retrieve client handles.

    Args:
        host: Host to bind server to.
        port: Port to bind server to.
        message_class: Base class for message types. Subclasses of the message type
            should have unique names. This argument is optional currently, but will be
            required in the future.
        http_server_root: Path to root for HTTP server.
        verbose: Toggle for print messages.
    """

    def __init__(
        self,
        host: str,
        port: int,
        message_class: Type[Message] = Message,
        http_server_root: Optional[Path] = None,
        verbose: bool = True,
    ):
        super().__init__()

        # Track connected clients.
        self._client_connect_cb: List[Callable[[ClientConnection], None]] = []
        self._client_disconnect_cb: List[Callable[[ClientConnection], None]] = []

        self._host = host
        self._port = port
        self._message_class = message_class
        self._http_server_root = http_server_root
        self._verbose = verbose

    def start(self) -> None:
        """Start the server."""

        # Start server thread.
        ready_sem = threading.Semaphore(value=1)
        ready_sem.acquire()
        threading.Thread(
            target=lambda: self._background_worker(ready_sem),
            daemon=True,
        ).start()

        # Wait for the thread to set self._event_loop and self._broadcast_buffer...
        ready_sem.acquire()

        # Broadcast buffer should be populated by the background worker.
        assert isinstance(self._broadcast_buffer, AsyncMessageBuffer)

    def on_client_connect(self, cb: Callable[[ClientConnection], Any]) -> None:
        """Attach a callback to run for newly connected clients."""
        self._client_connect_cb.append(cb)

    def on_client_disconnect(self, cb: Callable[[ClientConnection], Any]) -> None:
        """Attach a callback to run when clients disconnect."""
        self._client_disconnect_cb.append(cb)

    def broadcast(self, message: Message) -> None:
        """Pushes a message onto the broadcast queue. Message will be sent to all clients.

        Broadcasted messages are persistent: if a new client connects to the server,
        they will receive a buffered set of previously broadcasted messages. The buffer
        is culled using the value of `message.redundancy_key()`."""
        self._broadcast_buffer.push(message)

    def _background_worker(self, ready_sem: threading.Semaphore) -> None:
        host = self._host
        port = self._port
        message_class = self._message_class
        http_server_root = self._http_server_root

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

            if self._verbose:
                rich.print(
                    f"[bold](viser)[/bold] Connection opened ({client_id},"
                    f" {total_connections} total),"
                    f" {len(self._broadcast_buffer.stamped_message_from_id)} persistent"
                    " messages"
                )

            client_state = _ClientHandleState(
                message_buffer=asyncio.Queue(),
                event_loop=event_loop,
            )
            client_connection = ClientConnection(client_id, client_state)

            def handle_incoming(message: Message) -> None:
                threading.Thread(
                    target=lambda: self._handle_incoming_message(client_id, message)
                ).start()
                threading.Thread(
                    target=lambda: client_connection._handle_incoming_message(
                        client_id, message
                    )
                ).start()

            # New connection callbacks.
            for cb in self._client_connect_cb:
                cb(client_connection)

            try:
                # For each client: infinite loop over producers (which send messages)
                # and consumers (which receive messages).
                await asyncio.gather(
                    _single_connection_producer(
                        websocket, client_id, client_state.message_buffer
                    ),
                    _broadcast_producer(websocket, client_id, self._broadcast_buffer),
                    _consumer(websocket, handle_incoming, message_class),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # Disconnection callbacks.
                for cb in self._client_disconnect_cb:
                    cb(client_connection)

                # Cleanup.
                total_connections -= 1
                if self._verbose:
                    rich.print(
                        f"[bold](viser)[/bold] Connection closed ({client_id},"
                        f" {total_connections} total)"
                    )

        # Host client on the same port as the websocket.
        async def viser_http_server(
            path: str, request_headers: websockets.datastructures.Headers
        ) -> Optional[
            Tuple[http.HTTPStatus, websockets.datastructures.HeadersLike, bytes]
        ]:
            # Ignore websocket packets.
            if request_headers.get("Upgrade") == "websocket":
                return None

            # Strip out search params, get relative path.
            path = path.partition("?")[0]
            relpath = str(Path(path).relative_to("/"))
            if relpath == ".":
                relpath = "index.html"
            assert http_server_root is not None
            source = http_server_root / relpath

            # Try to read + send over file.
            try:
                return (
                    http.HTTPStatus.OK,
                    {
                        "content-type": str(
                            mimetypes.MimeTypes().guess_type(relpath)[0]
                        ),
                    },
                    source.read_bytes(),
                )
            except FileNotFoundError:
                return (http.HTTPStatus.NOT_FOUND, {}, b"404")  # type: ignore

        for _ in range(500):
            try:
                event_loop.run_until_complete(
                    websockets.server.serve(
                        serve,
                        host,
                        port,
                        compression=None,
                        process_request=(
                            viser_http_server if http_server_root is not None else None
                        ),
                    )
                )
                break
            except OSError:  # Port not available.
                port += 1
                continue

        if self._verbose:
            http_url = f"http://{host}:{port}"
            ws_url = f"ws://{host}:{port}"

            table = Table(
                title=None,
                show_header=False,
                box=box.MINIMAL,
                title_style=style.Style(bold=True),
            )
            if http_server_root is not None:
                table.add_row("HTTP", f"[link={http_url}]{http_url}[/link]")
            table.add_row("Websocket", f"[link={ws_url}]{ws_url}[/link]")

            rich.print(Panel(table, title="[bold]viser[/bold]", expand=False))

        event_loop.run_forever()


async def _single_connection_producer(
    websocket: WebSocketServerProtocol, client_id: ClientId, buffer: asyncio.Queue
) -> None:
    """Infinite loop to send messages from the client buffer."""
    while True:
        message = await buffer.get()
        if message.excluded_self_client == client_id:
            continue
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
    handle_message: Callable[[Message], None],
    message_class: Type[Message],
) -> None:
    """Infinite loop waiting for and then handling incoming messages."""
    while True:
        raw = await websocket.recv()
        assert isinstance(raw, bytes)
        message = message_class.deserialize(raw)
        handle_message(message)
