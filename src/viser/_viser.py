from __future__ import annotations

import contextlib
import dataclasses
import io
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import imageio.v3 as iio
import numpy as onp
import numpy.typing as npt
import rich
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Literal, override

from . import _client_autobuild, _messages, infra
from . import transforms as tf
from ._gui_api import GuiApi
from ._message_api import MessageApi, cast_vector
from ._scene_handles import FrameHandle, _SceneNodeHandleState
from ._tunnel import _ViserTunnel


@dataclasses.dataclass
class _CameraHandleState:
    """Information about a client's camera state."""

    client: ClientHandle
    wxyz: npt.NDArray[onp.float64]
    position: npt.NDArray[onp.float64]
    fov: float
    aspect: float
    look_at: npt.NDArray[onp.float64]
    up_direction: npt.NDArray[onp.float64]
    update_timestamp: float
    camera_cb: List[Callable[[CameraHandle], None]]


@dataclasses.dataclass
class CameraHandle:
    _state: _CameraHandleState

    @property
    def client(self) -> ClientHandle:
        """Client that this camera corresponds to."""
        return self._state.client

    @property
    def wxyz(self) -> npt.NDArray[onp.float64]:
        """Corresponds to the R in `P_world = [R | t] p_camera`. Synchronized
        automatically when assigned."""
        assert self._state.update_timestamp != 0.0
        return self._state.wxyz

    # Note: asymmetric properties are supported in Pyright, but not yet in mypy.
    # - https://github.com/python/mypy/issues/3004
    # - https://github.com/python/mypy/pull/11643
    @wxyz.setter
    def wxyz(self, wxyz: Tuple[float, float, float, float] | onp.ndarray) -> None:
        R_world_camera = tf.SO3(onp.asarray(wxyz)).as_matrix()
        look_distance = onp.linalg.norm(self.look_at - self.position)

        # We're following OpenCV conventions: look_direction is +Z, up_direction is -Y,
        # right_direction is +X.
        look_direction = R_world_camera[:, 2]
        up_direction = -R_world_camera[:, 1]
        right_direction = R_world_camera[:, 0]

        # Minimize our impact on the orbit controls by keeping the new up direction as
        # close to the old one as possible.
        projected_up_direction = (
            self.up_direction
            - float(self.up_direction @ right_direction) * right_direction
        )
        up_cosine = float(up_direction @ projected_up_direction)
        if abs(up_cosine) < 0.05:
            projected_up_direction = up_direction
        elif up_cosine < 0.0:
            projected_up_direction = up_direction

        new_look_at = look_direction * look_distance + self.position
        self.look_at = new_look_at
        self.up_direction = projected_up_direction
        self._state.wxyz = onp.asarray(wxyz)

    @property
    def position(self) -> npt.NDArray[onp.float64]:
        """Corresponds to the t in `P_world = [R | t] p_camera`. Synchronized
        automatically when assigned.

        The `look_at` point and `up_direction` vectors are maintained when updating
        `position`, which means that updates to `position` will often also affect `wxyz`.
        """
        assert self._state.update_timestamp != 0.0
        return self._state.position

    @position.setter
    def position(self, position: Tuple[float, float, float] | onp.ndarray) -> None:
        offset = onp.asarray(position) - onp.array(self.position)  # type: ignore
        self._state.position = onp.asarray(position)
        self.look_at = onp.array(self.look_at) + offset
        self._state.update_timestamp = time.time()
        self._state.client._queue(
            _messages.SetCameraPositionMessage(cast_vector(position, 3))
        )

    @property
    def fov(self) -> float:
        """Vertical field of view of the camera, in radians. Synchronized automatically
        when assigned."""
        assert self._state.update_timestamp != 0.0
        return self._state.fov

    @fov.setter
    def fov(self, fov: float) -> None:
        self._state.fov = fov
        self._state.update_timestamp = time.time()
        self._state.client._queue(_messages.SetCameraFovMessage(fov))

    @property
    def aspect(self) -> float:
        """Canvas width divided by height. Not assignable."""
        assert self._state.update_timestamp != 0.0
        return self._state.aspect

    @property
    def update_timestamp(self) -> float:
        assert self._state.update_timestamp != 0.0
        return self._state.update_timestamp

    @property
    def look_at(self) -> npt.NDArray[onp.float64]:
        """Look at point for the camera. Synchronized automatically when set."""
        assert self._state.update_timestamp != 0.0
        return self._state.look_at

    @look_at.setter
    def look_at(self, look_at: Tuple[float, float, float] | onp.ndarray) -> None:
        self._state.look_at = onp.asarray(look_at)
        self._state.update_timestamp = time.time()
        self._state.client._queue(
            _messages.SetCameraLookAtMessage(cast_vector(look_at, 3))
        )

    @property
    def up_direction(self) -> npt.NDArray[onp.float64]:
        """Up direction for the camera. Synchronized automatically when set."""
        assert self._state.update_timestamp != 0.0
        return self._state.up_direction

    @up_direction.setter
    def up_direction(
        self, up_direction: Tuple[float, float, float] | onp.ndarray
    ) -> None:
        self._state.up_direction = onp.asarray(up_direction)
        self._state.update_timestamp = time.time()
        self._state.client._queue(
            _messages.SetCameraUpDirectionMessage(cast_vector(up_direction, 3))
        )

    def on_update(
        self, callback: Callable[[CameraHandle], None]
    ) -> Callable[[CameraHandle], None]:
        """Attach a callback to run when a new camera message is received."""
        self._state.camera_cb.append(callback)
        return callback


@dataclasses.dataclass
class _ClientHandleState:
    server: infra.Server
    connection: infra.ClientConnection


@dataclasses.dataclass
class ClientHandle(MessageApi, GuiApi):
    """Handle for interacting with a specific client. Can be used to send messages to
    individual clients and read/write camera information."""

    client_id: int
    camera: CameraHandle
    _state: _ClientHandleState

    def __post_init__(self):
        super().__init__(self._state.connection)

    @override
    def _get_api(self) -> MessageApi:
        """Message API to use."""
        return self

    @override
    def _queue_unsafe(self, message: _messages.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.send(message)

    def get_render(
        self, height: int, width: int, transport_format: Literal["png", "jpeg"] = "jpeg"
    ) -> onp.ndarray:
        """Request a render from a client, block until it's done and received, then
        return it as a numpy array.

        Args:
            height: Height of rendered image. Should be <= the browser height.
            width: Width of rendered image. Should be <= the browser width.
            transport_format: Image transport format. JPEG will return a lossy (H, W, 3) RGB array. PNG will
                return a lossless (H, W, 4) RGBA array, but can cause memory issues on the frontend if called
                too quickly for higher-resolution images.
        """

        # Listen for a render reseponse message, which should contain the rendered
        # image.
        render_ready_event = threading.Event()
        out: Optional[onp.ndarray] = None

        def got_render_cb(
            client_id: int, message: _messages.GetRenderResponseMessage
        ) -> None:
            del client_id
            self._state.connection.unregister_handler(
                _messages.GetRenderResponseMessage, got_render_cb
            )
            nonlocal out
            out = iio.imread(
                io.BytesIO(message.payload),
                extension=f".{transport_format}",
            )
            render_ready_event.set()

        self._state.connection.register_handler(
            _messages.GetRenderResponseMessage, got_render_cb
        )
        self._queue(
            _messages.GetRenderRequestMessage(
                "image/jpeg" if transport_format == "jpeg" else "image/png",
                height=height,
                width=width,
                # Only used for JPEG. The main reason to use a lower quality version
                # value is (unfortunately) to make life easier for the Javascript
                # garbage collector.
                quality=80,
            )
        )
        render_ready_event.wait()
        assert out is not None
        return out

    @contextlib.contextmanager
    def atomic(self) -> Generator[None, None, None]:
        """Returns a context where:
        - No incoming messages, like camera or GUI state updates, are processed.
        - `viser` will attempt to group outgoing messages, which will then be sent after
          the context is exited.

        This can be helpful for things like animations, or when we want position and
        orientation updates to happen synchronously.
        """
        # If called multiple times in the same thread, we ignore inner calls.
        thread_id = threading.get_ident()
        if thread_id == self._locked_thread_id:
            got_lock = False
        else:
            self._atomic_lock.acquire()
            self._locked_thread_id = thread_id
            got_lock = True

        yield

        if got_lock:
            self._atomic_lock.release()
            self._locked_thread_id = -1


# We can serialize the state of a ViserServer via a tuple of
# (serialized message, timestamp) pairs.
SerializedServerState = Tuple[Tuple[bytes, float], ...]


@dataclasses.dataclass
class _ViserServerState:
    connection: infra.Server
    connected_clients: Dict[int, ClientHandle] = dataclasses.field(default_factory=dict)
    client_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class ViserServer(MessageApi, GuiApi):
    """Viser server class. The primary interface for functionality in `viser`.

    Commands on a server object (`add_frame`, `add_gui_*`, ...) will be sent to all
    clients, including new clients that connect after a command is called.

    Args:
        host: Host to bind server to.
        port: Port to bind server to.
        share: Experimental. If set to `True`, create and print a public, shareable URL
            for this instance of viser.
    """

    world_axes: FrameHandle
    """Handle for manipulating the world frame axes (/WorldAxes), which is instantiated
    and then hidden by default."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, share: bool = False):
        server = infra.Server(
            host=host,
            port=port,
            message_class=_messages.Message,
            http_server_root=Path(__file__).absolute().parent / "client" / "build",
            client_api_version=1,
        )
        self._server = server
        super().__init__(server)

        _client_autobuild.ensure_client_is_built()

        state = _ViserServerState(server)
        self._state = state
        self._client_connect_cb: List[Callable[[ClientHandle], None]] = []
        self._client_disconnect_cb: List[Callable[[ClientHandle], None]] = []

        # For new clients, register and add a handler for camera messages.
        @server.on_client_connect
        def _(conn: infra.ClientConnection) -> None:
            camera = CameraHandle(
                _CameraHandleState(
                    # TODO: values are initially not valid.
                    client=None,  # type: ignore
                    wxyz=onp.zeros(4),
                    position=onp.zeros(3),
                    fov=0.0,
                    aspect=0.0,
                    look_at=onp.zeros(3),
                    up_direction=onp.zeros(3),
                    update_timestamp=0.0,
                    camera_cb=[],
                )
            )
            client = ClientHandle(
                conn.client_id,
                camera,
                _ClientHandleState(server, conn),
            )
            camera._state.client = client
            first = True

            def handle_camera_message(
                client_id: infra.ClientId, message: _messages.ViewerCameraMessage
            ) -> None:
                nonlocal first

                assert client_id == client.client_id

                # Update the client's camera.
                with client._atomic_lock:
                    client.camera._state = _CameraHandleState(
                        client,
                        onp.array(message.wxyz),
                        onp.array(message.position),
                        message.fov,
                        message.aspect,
                        onp.array(message.look_at),
                        onp.array(message.up_direction),
                        time.time(),
                        camera_cb=client.camera._state.camera_cb,
                    )

                # We consider a client to be connected after the first camera message is
                # received.
                if first:
                    first = False
                    with self._state.client_lock:
                        state.connected_clients[conn.client_id] = client
                        for cb in self._client_connect_cb:
                            cb(client)

                for camera_cb in client.camera._state.camera_cb:
                    camera_cb(client.camera)

            conn.register_handler(_messages.ViewerCameraMessage, handle_camera_message)

        # Remove clients when they disconnect.
        @server.on_client_disconnect
        def _(conn: infra.ClientConnection) -> None:
            with self._state.client_lock:
                if conn.client_id not in state.connected_clients:
                    return

                handle = state.connected_clients.pop(conn.client_id)
                for cb in self._client_disconnect_cb:
                    cb(handle)

        # Start the server.
        server.start()

        # Form status print.
        port = server._port  # Port may have changed.
        http_url = f"http://{host}:{port}"
        ws_url = f"ws://{host}:{port}"
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("HTTP", http_url)
        table.add_row("Websocket", ws_url)

        # Create share tunnel if requested.
        if not share:
            self._share_tunnel = None
            rich.print(Panel(table, title="[bold]viser[/bold]", expand=False))
        else:
            rich.print("[bold](viser)[/bold] Share URL requested! (expires in 24 hours)")
            self._share_tunnel = _ViserTunnel(port)

            @self._share_tunnel.on_connect
            def _() -> None:
                assert self._share_tunnel is not None
                share_url = self._share_tunnel.get_url()
                if share_url is None:
                    rich.print("[bold](viser)[/bold] Could not generate share URL")
                else:
                    table.add_row("Share URL", share_url)
                rich.print(Panel(table, title="[bold]viser[/bold]", expand=False))

        self.reset_scene()
        self.world_axes = FrameHandle(
            _SceneNodeHandleState(
                "/WorldAxes",
                self,
                wxyz=onp.array([1.0, 0.0, 0.0, 0.0]),
                position=onp.zeros(3),
            )
        )
        self.world_axes.visible = False

    def stop(self) -> None:
        """Stop the Viser server and associated threads and tunnels."""
        self._server.stop()
        if self._share_tunnel is not None:
            self._share_tunnel.close()

    @override
    def _get_api(self) -> MessageApi:
        """Message API to use."""
        return self

    @override
    def _queue_unsafe(self, message: _messages.Message) -> None:
        """Define how the message API should send messages."""
        self._server.broadcast(message)

    def get_clients(self) -> Dict[int, ClientHandle]:
        """Creates and returns a copy of the mapping from connected client IDs to
        handles."""
        with self._state.client_lock:
            return self._state.connected_clients.copy()

    def on_client_connect(
        self, cb: Callable[[ClientHandle], None]
    ) -> Callable[[ClientHandle], None]:
        """Attach a callback to run for newly connected clients."""
        with self._state.client_lock:
            clients = self._state.connected_clients.copy().values()
            self._client_connect_cb.append(cb)

        # Trigger callback on any already-connected clients.
        # If we have:
        #
        #     server = viser.ViserServer()
        #     server.on_client_connect(...)
        #
        # This makes sure that the the callback is applied to any clients that
        # connect between the two lines.
        for client in clients:
            cb(client)
        return cb

    def on_client_disconnect(
        self, cb: Callable[[ClientHandle], None]
    ) -> Callable[[ClientHandle], None]:
        """Attach a callback to run when clients disconnect."""
        self._client_disconnect_cb.append(cb)
        return cb

    @contextlib.contextmanager
    def atomic(self) -> Generator[None, None, None]:
        """Returns a context where:
        - All outgoing messages are grouped and applied by clients atomically.
        - No incoming messages, like camera or GUI state updates, are processed.

        This can be helpful for things like animations, or when we want position and
        orientation updates to happen synchronously.
        """
        # Acquire the global atomic lock.
        # If called multiple times in the same thread, we ignore inner calls.
        thread_id = threading.get_ident()
        if thread_id == self._locked_thread_id:
            got_lock = False
        else:
            self._atomic_lock.acquire()
            self._locked_thread_id = thread_id
            got_lock = True

        with contextlib.ExitStack() as stack:
            if got_lock:
                # Grab each client's atomic lock.
                # We don't need to do anything with `client._locked_thread_id`.
                for client in self.get_clients().values():
                    stack.enter_context(client._atomic_lock)

            yield

        if got_lock:
            self._atomic_lock.release()
            self._locked_thread_id = -1
