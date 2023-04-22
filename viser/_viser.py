from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from typing_extensions import override

from . import infra
from ._message_api import MessageApi
from ._messages import (
    SetCameraFovMessage,
    SetCameraOrientationMessage,
    SetCameraPositionMessage,
    ViewerCameraMessage,
)
from ._scene_handle import SceneNodeHandle, _SceneNodeHandleState


@dataclasses.dataclass
class _CameraHandleState:
    """Information about a client's camera state."""

    connection: infra.ClientConnection
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    fov: float
    aspect: float
    update_timestamp: float


@dataclasses.dataclass
class CameraHandle:
    _state: _CameraHandleState

    @property
    def wxyz(self) -> Tuple[float, float, float, float]:
        """Corresponds to the R in `P_world = [R | t] p_camera`. Synchronized
        automatically when assigned."""
        return self._state.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: Tuple[float, float, float, float]) -> None:
        self._state.wxyz = wxyz
        self._state.update_timestamp = time.time()
        self._state.connection.send(SetCameraOrientationMessage(wxyz))

    @property
    def position(self) -> Tuple[float, float, float]:
        """Corresponds to the t in `P_world = [R | t] p_camera`. Synchronized
        automatically when assigned."""
        return self._state.position

    @position.setter
    def position(self, position: Tuple[float, float, float]) -> None:
        self._state.position = position
        self._state.update_timestamp = time.time()
        self._state.connection.send(SetCameraPositionMessage(position))

    @property
    def fov(self) -> float:
        """Vertical field of view of the camera, in radians. Synchronized automatically
        when assigned."""
        return self._state.fov

    @fov.setter
    def fov(self, fov: float) -> None:
        self._state.fov = fov
        self._state.update_timestamp = time.time()
        self._state.connection.send(SetCameraFovMessage(fov))

    @property
    def aspect(self) -> float:
        """Canvas width divided by height. Not assignable."""
        return self._state.aspect

    @property
    def update_timestamp(self) -> float:
        return self._state.update_timestamp


@dataclasses.dataclass
class _ClientHandleState:
    connection: infra.ClientConnection
    camera_cb: List[Callable[[ClientHandle], None]]


@dataclasses.dataclass
class ClientHandle(MessageApi):
    """Handle for interacting with a specific client. Can be used to send messages to
    individual clients, read camera information, etc."""

    client_id: int
    camera: CameraHandle
    _state: _ClientHandleState

    def __post_init__(self):
        super().__init__(self._state.connection)

    @override
    def _queue(self, message: infra.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.send(message)

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
            camera = CameraHandle(
                _CameraHandleState(
                    # TODO: values are initially not valid.
                    conn,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, 0.0, 0.0),
                    fov=0.0,
                    aspect=0.0,
                    update_timestamp=0.0,
                )
            )
            client = ClientHandle(conn.client_id, camera, _ClientHandleState(conn, []))

            for cb in self._client_connect_cb:
                cb(client)

            def handle_camera_message(
                client_id: infra.ClientId, message: ViewerCameraMessage
            ) -> None:
                assert client_id == client.client_id
                client.camera._state = _CameraHandleState(
                    conn,
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
