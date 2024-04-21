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
from ._tunnel import ViserTunnel


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

        # Update lookat and up direction.
        self.look_at = new_look_at
        self.up_direction = projected_up_direction

        # The internal camera orientation should be set in the look_at /
        # up_direction setters. We can uncomment this assert to check this.
        # assert onp.allclose(self._state.wxyz, wxyz) or onp.allclose(
        #     self._state.wxyz, -wxyz
        # )

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

    def _update_wxyz(self) -> None:
        """Compute and update the camera orientation from the internal look_at, position, and up vectors."""
        z = self._state.look_at - self._state.position
        z /= onp.linalg.norm(z)
        y = tf.SO3.exp(z * onp.pi) @ self._state.up_direction
        y = y - onp.dot(z, y) * z
        y /= onp.linalg.norm(y)
        x = onp.cross(y, z)
        self._state.wxyz = tf.SO3.from_matrix(onp.stack([x, y, z], axis=1)).wxyz

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
        self._update_wxyz()
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
        self._update_wxyz()
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

        connection = self.client._state.connection

        def got_render_cb(
            client_id: int, message: _messages.GetRenderResponseMessage
        ) -> None:
            del client_id
            connection.unregister_handler(
                _messages.GetRenderResponseMessage, got_render_cb
            )
            nonlocal out
            out = iio.imread(
                io.BytesIO(message.payload),
                extension=f".{transport_format}",
            )
            render_ready_event.set()

        connection.register_handler(_messages.GetRenderResponseMessage, got_render_cb)
        self.client._queue(
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


@dataclasses.dataclass
class _ClientHandleState:
    viser_server: ViserServer
    server: infra.Server
    connection: infra.ClientConnection


@dataclasses.dataclass
class ClientHandle(MessageApi, GuiApi):
    """Handle for interacting with a specific client. Can be used to send messages to
    individual clients and read/write camera information."""

    client_id: int
    """Unique ID for this client."""
    camera: CameraHandle
    """Handle for reading from and manipulating the client's viewport camera."""
    _state: _ClientHandleState

    def __post_init__(self):
        super().__init__(self._state.connection, self._state.server._thread_executor)

    @override
    def _get_api(self) -> MessageApi:
        """Message API to use."""
        return self

    @override
    def _queue_unsafe(self, message: _messages.Message) -> None:
        """Define how the message API should send messages."""
        self._state.connection.send(message)

    @override
    @contextlib.contextmanager
    def atomic(self) -> Generator[None, None, None]:
        """Returns a context where: all outgoing messages are grouped and applied by
        clients atomically.

        This should be treated as a soft constraint that's helpful for things
        like animations, or when we want position and orientation updates to
        happen synchronously.

        Returns:
            Context manager.
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

    @override
    def flush(self) -> None:
        """Flush the outgoing message buffer. Any buffered messages will immediately be
        sent. (by default they are windowed)"""
        self._state.server.flush_client(self.client_id)


# We can serialize the state of a ViserServer via a tuple of
# (serialized message, timestamp) pairs.
SerializedServerState = Tuple[Tuple[bytes, float], ...]


def dummy_process() -> None:
    pass


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
        label: Label shown at the top of the GUI panel.
    """

    world_axes: FrameHandle
    """Handle for manipulating the world frame axes (/WorldAxes), which is instantiated
    and then hidden by default."""

    # Hide deprecated arguments from docstring and type checkers.
    def __init__(
        self, host: str = "0.0.0.0", port: int = 8080, label: Optional[str] = None
    ):
        ...

    def _actual_init(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: Optional[str] = None,
        **_deprecated_kwargs,
    ):
        # Create server.
        server = infra.Server(
            host=host,
            port=port,
            message_class=_messages.Message,
            http_server_root=Path(__file__).absolute().parent / "client" / "build",
            client_api_version=1,
        )
        self._server = server
        super().__init__(server, server._thread_executor)

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
                _ClientHandleState(self, server, conn),
            )
            camera._state.client = client
            first = True

            def handle_camera_message(
                client_id: infra.ClientId, message: _messages.ViewerCameraMessage
            ) -> None:
                nonlocal first

                assert client_id == client.client_id

                # Update the client's camera.
                with client.atomic():
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

        server.register_handler(
            _messages.ShareUrlDisconnect,
            lambda client_id, msg: self.disconnect_share_url(),
        )
        server.register_handler(
            _messages.ShareUrlRequest, lambda client_id, msg: self.request_share_url()
        )

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
        rich.print(Panel(table, title="[bold]viser[/bold]", expand=False))

        self._share_tunnel: Optional[ViserTunnel] = None

        # Create share tunnel if requested.
        # This is deprecated: we should use get_share_url() instead.
        share = _deprecated_kwargs.get("share", False)
        if share:
            self.request_share_url()

        self.reset_scene()
        self.set_gui_panel_label(label)

        # Create a handle for the world axes, which are hardcoded to exist in the client.
        self.world_axes = FrameHandle(
            _SceneNodeHandleState(
                "/WorldAxes",
                self,
                wxyz=onp.array([1.0, 0.0, 0.0, 0.0]),
                position=onp.zeros(3),
            )
        )
        self.world_axes.visible = False

    def get_host(self) -> str:
        """Returns the host address of the Viser server.

        Returns:
            Host address as string.
        """
        return self._server._host

    def get_port(self) -> int:
        """Returns the port of the Viser server. This could be different from the
        originally requested one.

        Returns:
            Port as integer.
        """
        return self._server._port

    def request_share_url(self, verbose: bool = True) -> Optional[str]:
        """Request a share URL for the Viser server, which allows for public access.
        On the first call, will block until a connecting with the share URL server is
        established. Afterwards, the URL will be returned directly.

        This is an experimental feature that relies on an external server; it shouldn't
        be relied on for critical applications.

        Returns:
            Share URL as string, or None if connection fails or is closed.
        """
        if self._share_tunnel is not None:
            # Tunnel already exists.
            while self._share_tunnel.get_status() in ("ready", "connecting"):
                time.sleep(0.05)
            return self._share_tunnel.get_url()
        else:
            # Create a new tunnel!.
            if verbose:
                rich.print("[bold](viser)[/bold] Share URL requested!")

            connect_event = threading.Event()

            self._share_tunnel = ViserTunnel("share.viser.studio", self._server._port)

            @self._share_tunnel.on_disconnect
            def _() -> None:
                rich.print("[bold](viser)[/bold] Disconnected from share URL")
                self._share_tunnel = None
                self._server.broadcast(_messages.ShareUrlUpdated(None))

            @self._share_tunnel.on_connect
            def _(max_clients: int) -> None:
                assert self._share_tunnel is not None
                share_url = self._share_tunnel.get_url()
                if verbose:
                    if share_url is None:
                        rich.print("[bold](viser)[/bold] Could not generate share URL")
                    else:
                        rich.print(
                            f"[bold](viser)[/bold] Generated share URL (expires in 24 hours, max {max_clients} clients): {share_url}"
                        )
                self._server.broadcast(_messages.ShareUrlUpdated(share_url))
                connect_event.set()

            connect_event.wait()

            url = self._share_tunnel.get_url()
            return url

    def disconnect_share_url(self) -> None:
        """Disconnect from the share URL server."""
        if self._share_tunnel is not None:
            self._share_tunnel.close()
        else:
            rich.print(
                "[bold](viser)[/bold] Tried to disconnect from share URL, but already disconnected"
            )

    def stop(self) -> None:
        """Stop the Viser server and associated threads and tunnels."""
        self._server.stop()
        if self._share_tunnel is not None:
            self._share_tunnel.close()

    def get_clients(self) -> Dict[int, ClientHandle]:
        """Creates and returns a copy of the mapping from connected client IDs to
        handles.

        Returns:
            Dictionary of clients.
        """
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

    @override
    @contextlib.contextmanager
    def atomic(self) -> Generator[None, None, None]:
        """Returns a context where: all outgoing messages are grouped and applied by
        clients atomically.

        This should be treated as a soft constraint that's helpful for things
        like animations, or when we want position and orientation updates to
        happen synchronously.

        Returns:
            Context manager.
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

    @override
    def flush(self) -> None:
        """Flush the outgoing message buffer. Any buffered messages will immediately be
        sent. (by default they are windowed)"""
        self._server.flush()

    @override
    def _get_api(self) -> MessageApi:
        """Message API to use."""
        return self

    @override
    def _queue_unsafe(self, message: _messages.Message) -> None:
        """Define how the message API should send messages."""
        self._server.broadcast(message)


ViserServer.__init__ = ViserServer._actual_init  # type: ignore
