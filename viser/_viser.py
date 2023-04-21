from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from typing_extensions import override

from . import infra
from ._message_api import MessageApi
from ._messages import ViewerCameraMessage
from ._scene_handle import SceneNodeHandle, _SceneNodeHandleState


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

    client_id: int
    _state: _ClientHandleState

    def __post_init__(self):
        super().__init__(self._state.connection)

    @override
    def _queue(self, message: infra.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.send(message)

    @property
    def camera(self) -> CameraState:
        """Get the view camera from a particular client. Blocks if not available yet."""
        # TODO: there's a risk of getting stuck in an infinite loop here.
        while self._state.camera_info is None:
            time.sleep(0.01)
        return self._state.camera_info

    @camera.setter
    def camera(self, camera: CameraState) -> None:
        # TODO
        raise NotImplementedError()

    def on_camera_update(
        self, callback: Callable[[ClientHandle], None]
    ) -> Callable[[ClientHandle], None]:
        """Attach a callback to run when a new camera message is received."""
        self._state.camera_cb.append(callback)
        return callback


@dataclasses.dataclass
class _ViserServerState:
    connection: infra.Server
    connected_clients: Dict[int, ClientHandle]
    client_lock: threading.Lock


class ViserServer(MessageApi):
    """Viser server class. The primary interface for functionality in `viser`."""

    world_axes: SceneNodeHandle
    """Handle for manipulating the world frame axes (/WorldAxes), which is instantiated
    and then hidden by default."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        server = infra.Server(
            host=host,
            port=port,
            http_server_root=Path(__file__).absolute().parent / "client" / "build",
        )
        super().__init__(server)

        state = _ViserServerState(server, {}, threading.Lock())
        self._state = state
        self._client_connect_cb: List[Callable[[ClientHandle], None]] = []
        self._client_disconnect_cb: List[Callable[[ClientHandle], None]] = []

        # For new clients, register and add a handler for camera messages.
        @server.on_client_connect
        def _(conn: infra.ClientConnection) -> None:
            client = ClientHandle(conn.client_id, _ClientHandleState(conn, None, []))

            for cb in self._client_connect_cb:
                cb(client)

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
                handle = state.connected_clients.pop(conn.client_id)

            for cb in self._client_disconnect_cb:
                cb(handle)

        # Start the server.
        server.start()
        self.reset_scene()
        self.world_axes = SceneNodeHandle(_SceneNodeHandleState("/WorldAxes", self))
        self.world_axes.visible = False

    def get_clients(self) -> Dict[int, ClientHandle]:
        """Creates and returns a copy of the mapping from connected client IDs to
        handles."""
        with self._state.client_lock:
            return self._state.connected_clients.copy()

    def on_client_connect(self, cb: Callable[[ClientHandle], Any]) -> None:
        """Attach a callback to run for newly connected clients."""
        self._client_connect_cb.append(cb)

    def on_client_disconnect(self, cb: Callable[[ClientHandle], Any]) -> None:
        """Attach a callback to run when clients disconnect."""
        self._client_disconnect_cb.append(cb)

    @override
    def _queue(self, message: infra.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.broadcast(message)
