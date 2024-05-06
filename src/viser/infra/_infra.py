from __future__ import annotations

import asyncio
import dataclasses
import http
import mimetypes
import threading
from asyncio.events import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Type, TypeVar

import msgpack
import rich
import websockets.connection
import websockets.datastructures
import websockets.exceptions
import websockets.server
from typing_extensions import Literal, assert_never
from websockets.legacy.server import WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._messages import Message


@dataclasses.dataclass
class _ClientHandleState:
    # Internal state for ClientConnection objects.
    # message_buffer: asyncio.Queue
    message_buffer: AsyncMessageBuffer
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
        callback: Callable[[ClientId, TMessage], Any],
    ) -> None:
        """Register a handler for a particular message type."""
        if message_cls not in self._incoming_handlers:
            self._incoming_handlers[message_cls] = []
        self._incoming_handlers[message_cls].append(callback)  # type: ignore

    def unregister_handler(
        self,
        message_cls: Type[TMessage],
        callback: Optional[Callable[[ClientId, TMessage], Any]] = None,
    ):
        """Unregister a handler for a particular message type."""
        assert (
            message_cls in self._incoming_handlers
        ), "Tried to unregister a handler that hasn't been registered."
        if callback is None:
            self._incoming_handlers.pop(message_cls)
        else:
            self._incoming_handlers[message_cls].remove(callback)  # type: ignore

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
        self._state.message_buffer.push(message)


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
        client_api_version: Flag for backwards compatibility. 0 sends individual
            messages. 1 sends windowed messages.
    """

    def __init__(
        self,
        host: str,
        port: int,
        message_class: Type[Message] = Message,
        http_server_root: Optional[Path] = None,
        verbose: bool = True,
        client_api_version: Literal[0, 1] = 0,
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
        self._client_api_version: Literal[0, 1] = client_api_version

        self._thread_executor = ThreadPoolExecutor(max_workers=32)
        self._shutdown_event = threading.Event()

        self._client_state_from_id: Dict[int, _ClientHandleState] = {}

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

    def stop(self) -> None:
        """Stop the server."""
        self._thread_executor.shutdown(wait=True)
        self._event_loop.stop()

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

    def flush(self) -> None:
        """Flush the outgoing message buffer for broadcasted messages. Any buffered
        messages will immediately be sent. (by default they are windowed)"""
        # TODO: we should add a flush event.
        self._broadcast_buffer.flush()

    def flush_client(self, client_id: int) -> None:
        """Flush the outgoing message buffer for a particular client. Any buffered
        messages will immediately be sent. (by default they are windowed)"""
        self._client_state_from_id[client_id].message_buffer.flush()

    def _background_worker(self, ready_sem: threading.Semaphore) -> None:
        host = self._host
        port = self._port
        message_class = self._message_class
        http_server_root = self._http_server_root

        # Need to make a new event loop for notebook compatbility.
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        self._event_loop = event_loop
        self._broadcast_buffer = AsyncMessageBuffer(
            event_loop, persistent_messages=True
        )

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
                    f" {len(self._broadcast_buffer.message_from_id)} persistent"
                    " messages"
                )

            client_state = _ClientHandleState(
                AsyncMessageBuffer(event_loop, persistent_messages=False),
                event_loop,
            )
            client_connection = ClientConnection(client_id, client_state)
            self._client_state_from_id[client_id] = client_state

            def handle_incoming(message: Message) -> None:
                self._thread_executor.submit(
                    error_print_wrapper(
                        lambda: self._handle_incoming_message(client_id, message)
                    )
                )
                self._thread_executor.submit(
                    error_print_wrapper(
                        lambda: client_connection._handle_incoming_message(
                            client_id, message
                        )
                    )
                )

            # New connection callbacks.
            for cb in self._client_connect_cb:
                cb(client_connection)

            try:
                # For each client: infinite loop over producers (which send messages)
                # and consumers (which receive messages).
                await asyncio.gather(
                    _message_producer(
                        websocket,
                        client_state.message_buffer,
                        client_id,
                        self._client_api_version,
                    ),
                    _message_producer(
                        websocket,
                        self._broadcast_buffer,
                        client_id,
                        self._client_api_version,
                    ),
                    _message_consumer(websocket, handle_incoming, message_class),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # We use a sentinel value to signal that the client producer thread
                # should exit.
                #
                # This is partially cosmetic: it allows us to safely finish pending
                # queue get() tasks, which suppresses a "Task was destroyed but it is
                # pending" error.
                client_state.message_buffer.set_done()

                # Disconnection callbacks.
                for cb in self._client_disconnect_cb:
                    cb(client_connection)

                # Cleanup.
                self._client_state_from_id.pop(client_id)
                total_connections -= 1
                if self._verbose:
                    rich.print(
                        f"[bold](viser)[/bold] Connection closed ({client_id},"
                        f" {total_connections} total)"
                    )

        # Host client on the same port as the websocket.
        file_cache: Dict[Path, bytes] = {}
        file_cache_gzipped: Dict[Path, bytes] = {}

        import gzip

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

            source_path = http_server_root / relpath
            if not source_path.exists():
                return (http.HTTPStatus.NOT_FOUND, {}, b"404")  # type: ignore

            use_gzip = "gzip" in request_headers.get("Accept-Encoding", "")

            response_headers = {
                "Content-Type": str(mimetypes.MimeTypes().guess_type(relpath)[0]),
            }

            if source_path not in file_cache:
                file_cache[source_path] = source_path.read_bytes()
            if use_gzip:
                response_headers["Content-Encoding"] = "gzip"
                if source_path not in file_cache_gzipped:
                    file_cache_gzipped[source_path] = gzip.compress(
                        file_cache[source_path]
                    )
                response_payload = file_cache_gzipped[source_path]
            else:
                response_payload = file_cache[source_path]

            # Try to read + send over file.
            return (http.HTTPStatus.OK, response_headers, response_payload)

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

        self._port = port
        ready_sem.release()
        event_loop.run_forever()
        rich.print("[bold](viser)[/bold] Server stopped")


async def _message_producer(
    websocket: WebSocketServerProtocol,
    buffer: AsyncMessageBuffer,
    client_id: int,
    client_api_version: Literal[0, 1],
) -> None:
    """Infinite loop to broadcast windows of messages from a buffer."""
    window_generator = buffer.window_generator(client_id)
    while not buffer.done:
        outgoing = await window_generator.__anext__()
        if client_api_version == 1:
            serialized = msgpack.packb(
                tuple(message.as_serializable_dict() for message in outgoing)
            )
            assert isinstance(serialized, bytes)
            await websocket.send(serialized)
        elif client_api_version == 0:
            for msg in outgoing:
                serialized = msgpack.packb(msg.as_serializable_dict())
                assert isinstance(serialized, bytes)
                await websocket.send(serialized)
        else:
            assert_never(client_api_version)


async def _message_consumer(
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


def error_print_wrapper(inner: Callable[[], Any]) -> Callable[[], None]:
    """Wrap a Callable to print error messages when they happen.

    This can be helpful for jobs submitted to ThreadPoolExecutor instances, which, by
    default, will suppress error messages until returned futures are awaited.
    """

    def wrapped() -> None:
        try:
            inner()
        except Exception as e:
            import traceback as tb

            tb.print_exception(type(e), e, e.__traceback__, limit=100)

    return wrapped
