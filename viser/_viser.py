from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from typing_extensions import override

from . import infra
from ._message_api import MessageApi
from ._messages import ViewerCameraMessage


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
    connection: infra.ClientConnection
    camera_info: Optional[CameraState]
    camera_cb: List[Callable[[ClientHandle], None]]


@dataclasses.dataclass
class ClientHandle(MessageApi):
    """Handle for interacting with a specific client. Can be used to send messages to
    individual clients, read camera information, etc."""

    client_id: infra.ClientId
    _state: _ClientHandleState

    def __post_init__(self):
        super().__init__(self._state.connection)

    @override
    def _queue(self, message: infra.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.queue_message(message)

    def get_camera(self) -> CameraState:
        """Get the view camera from a particular client. Blocks if not available yet."""
        # TODO: there's a risk of getting stuck in an infinite loop here.
        while self._state.camera_info is None:
            time.sleep(0.01)
        return self._state.camera_info

    def on_camera_update(
        self, callback: Callable[[ClientHandle], None]
    ) -> Callable[[ClientHandle], None]:
        """Attach a callback to run when a new camera message is received."""
        self._state.camera_cb.append(callback)
        return callback


@dataclasses.dataclass
class _ViserServerState:
    connection: infra.Server
    connected_clients: Dict[infra.ClientId, ClientHandle]
    client_lock: threading.Lock


class ViserServer(MessageApi):
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        server = infra.Server(
            host=host,
            port=port,
            http_server_root=Path(__file__).absolute().parent / "client" / "build",
        )
        super().__init__(server)

        state = _ViserServerState(server, {}, threading.Lock())
        self._state = state

        # For new clients, register and add a handler for camera messages.
        @server.on_client_connect
        def _(conn: infra.ClientConnection) -> None:
            client = ClientHandle(conn.client_id, _ClientHandleState(conn, None, []))

            def handle_camera_message(
                client_id: infra.ClientId, message: ViewerCameraMessage
            ) -> None:
                assert client_id == client.client_id
                client._state.camera_info = CameraState(
                    message.wxyz,
                    message.position,
                    message.fov,
                    message.aspect,
                    time.time(),
                )
                for cb in client._state.camera_cb:
                    cb(client)

            with self._state.client_lock:
                state.connected_clients[conn.client_id] = client
            conn.register_handler(ViewerCameraMessage, handle_camera_message)

        # Remove clients when they disconnect.
        @server.on_client_disconnect
        def _(conn: infra.ClientConnection) -> None:
            with self._state.client_lock:
                state.connected_clients.pop(conn.client_id)

        # Start the server.
        server.start()
        self.reset_scene()

    def get_clients(self) -> Dict[infra.ClientId, ClientHandle]:
        """Get connected clients."""
        with self._state.client_lock:
            return self._state.connected_clients.copy()

    @override
    def _queue(self, message: infra.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.broadcast(message)
