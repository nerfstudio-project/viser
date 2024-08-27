from __future__ import annotations

import dataclasses
import io
import mimetypes
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager

import imageio.v3 as iio
import numpy as onp
import numpy.typing as npt
import rich
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Literal

from . import _client_autobuild, _messages, infra
from . import transforms as tf
from ._gui_api import GuiApi, _make_unique_id
from ._gui_handles import GuiNotificationHandle
from ._scene_api import SceneApi, cast_vector
from ._tunnel import ViserTunnel
from .infra._infra import RecordHandle


class _BackwardsCompatibilityShim:
    """Shims for backward compatibility with viser API from version
    `<=0.1.30`."""

    def __getattr__(self, name: str) -> Any:
        fixed_name = {
            # Map from old method names (viser v0.1.*) to new methods names.
            "reset_scene": "reset",
            "set_global_scene_node_visibility": "set_global_visibility",
            "on_scene_pointer": "on_pointer_event",
            "on_scene_pointer_removed": "on_pointer_callback_removed",
            "remove_scene_pointer_callback": "remove_pointer_callback",
            "add_mesh": "add_mesh_simple",
        }.get(name, name)
        if hasattr(self.scene, fixed_name):
            warnings.warn(
                f"{type(self).__name__}.{name} has been deprecated, use {type(self).__name__}.scene.{fixed_name} instead. Alternatively, pin to `viser<0.2.0`.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return object.__getattribute__(self.scene, fixed_name)

        fixed_name = name.replace("add_gui_", "add_").replace("set_gui_", "set_")
        if hasattr(self.gui, fixed_name):
            warnings.warn(
                f"{type(self).__name__}.{name} has been deprecated, use {type(self).__name__}.gui.{fixed_name} instead. Alternatively, pin to `viser<0.2.0`.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return object.__getattribute__(self.gui, fixed_name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


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
    camera_cb: list[Callable[[CameraHandle], None]]


class CameraHandle:
    """A handle for reading and writing the camera state of a particular
    client. Typically accessed via :attr:`ClientHandle.camera`."""

    def __init__(self, client: ClientHandle) -> None:
        self._state = _CameraHandleState(
            client,
            wxyz=onp.zeros(4),
            position=onp.zeros(3),
            fov=0.0,
            aspect=0.0,
            look_at=onp.zeros(3),
            up_direction=onp.zeros(3),
            update_timestamp=0.0,
            camera_cb=[],
        )

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
    def wxyz(self, wxyz: tuple[float, float, float, float] | onp.ndarray) -> None:
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
    def position(self, position: tuple[float, float, float] | onp.ndarray) -> None:
        offset = onp.asarray(position) - onp.array(self.position)  # type: ignore
        self._state.position = onp.asarray(position)
        self.look_at = onp.array(self.look_at) + offset
        self._state.update_timestamp = time.time()
        self._state.client._websock_connection.queue_message(
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
        self._state.client._websock_connection.queue_message(
            _messages.SetCameraFovMessage(fov)
        )

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
    def look_at(self, look_at: tuple[float, float, float] | onp.ndarray) -> None:
        self._state.look_at = onp.asarray(look_at)
        self._state.update_timestamp = time.time()
        self._update_wxyz()
        self._state.client._websock_connection.queue_message(
            _messages.SetCameraLookAtMessage(cast_vector(look_at, 3))
        )

    @property
    def up_direction(self) -> npt.NDArray[onp.float64]:
        """Up direction for the camera. Synchronized automatically when set."""
        assert self._state.update_timestamp != 0.0
        return self._state.up_direction

    @up_direction.setter
    def up_direction(
        self, up_direction: tuple[float, float, float] | onp.ndarray
    ) -> None:
        self._state.up_direction = onp.asarray(up_direction)
        self._update_wxyz()
        self._state.update_timestamp = time.time()
        self._state.client._websock_connection.queue_message(
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
        out: onp.ndarray | None = None

        connection = self.client._websock_connection

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
        self.client._websock_connection.queue_message(
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


# Don't inherit from _BackwardsCompatibilityShim during type checking, because
# this will unnecessarily suppress type errors. (from the overriding of
# __getattr__).
class ClientHandle(_BackwardsCompatibilityShim if not TYPE_CHECKING else object):
    """A handle is created for each client that connects to a server. Handles can be
    used to communicate with just one client, as well as for reading and writing of
    camera state.

    Similar to :class:`ViserServer`, client handles also expose scene and GUI
    interfaces at :attr:`ClientHandle.scene` and :attr:`ClientHandle.gui`. If
    these are used, for example via a client's
    :meth:`SceneApi.add_point_cloud()` method, created elements are local to
    only one specific client.
    """

    def __init__(
        self, conn: infra.WebsockClientConnection, server: ViserServer
    ) -> None:
        # Private attributes.
        self._websock_connection = conn
        self._viser_server = server

        # Public attributes.
        self.scene: SceneApi = SceneApi(
            self, thread_executor=server._websock_server._thread_executor
        )
        """Handle for interacting with the 3D scene."""
        self.gui: GuiApi = GuiApi(
            self, thread_executor=server._websock_server._thread_executor
        )
        """Handle for interacting with the GUI."""
        self.client_id: int = conn.client_id
        """Unique ID for this client."""
        self.camera: CameraHandle = CameraHandle(self)
        """Handle for reading from and manipulating the client's viewport camera."""

    def flush(self) -> None:
        """Flush the outgoing message buffer. Any buffered messages will immediately be
        sent. (by default they are windowed)"""
        self._viser_server._websock_server.flush_client(self.client_id)

    def atomic(self) -> ContextManager[None]:
        """Returns a context where: all outgoing messages are grouped and applied by
        clients atomically.

        This should be treated as a soft constraint that's helpful for things
        like animations, or when we want position and orientation updates to
        happen synchronously.

        Returns:
            Context manager.
        """
        return self._websock_connection.atomic()

    def send_file_download(
        self, filename: str, content: bytes, chunk_size: int = 1024 * 1024
    ) -> None:
        """Send a file for a client or clients to download.

        Args:
            filename: Name of the file to send. Used to infer MIME type.
            content: Content of the file.
            chunk_size: Number of bytes to send at a time.
        """
        mime_type = mimetypes.guess_type(filename, strict=False)[0]
        if mime_type is None:
            mime_type = "application/octet-stream"

        parts = [
            content[i * chunk_size : (i + 1) * chunk_size]
            for i in range(int(onp.ceil(len(content) / chunk_size)))
        ]

        uuid = _make_unique_id()
        self._websock_connection.queue_message(
            _messages.FileTransferStart(
                source_component_id=None,
                transfer_uuid=uuid,
                filename=filename,
                mime_type=mime_type,
                part_count=len(parts),
                size_bytes=len(content),
            )
        )
        for i, part in enumerate(parts):
            self._websock_connection.queue_message(
                _messages.FileTransferPart(
                    None,
                    transfer_uuid=uuid,
                    part=i,
                    content=part,
                )
            )
            self.flush()

    def add_notification(
        self,
        title: str,
        body: str,
        loading: bool = False,
        with_close_button: bool = True,
        auto_close: int | Literal[False] = False,
        order: float | None = None,
    ) -> GuiNotificationHandle:
        """Add a notification, which can be toggled on/off in the GUI.

        Args:
            title: Title to display on the notification.
            body: Message to display on the notification body.
            loading: Whether the notification shows loading icon.
            with_close_button: Whether the notification can be manually closed.
            auto_close: Time in ms before the notification automatically closes;
                        otherwise False such that the notification never closes on its own.

        Returns:
            A handle that can be used to interact with the GUI element.
        """
        handle = GuiNotificationHandle(
            _gui_api=self.gui,
            _id=_make_unique_id(),
            _title=title,
            _body=body,
            _loading=loading,
            _with_close_button=with_close_button,
            _auto_close=auto_close,
        )
        self.gui._websock_interface.queue_message(
            _messages.NotificationMessage(
                id=handle._id,
                title=title,
                body=body,
                loading=loading,
                with_close_button=with_close_button,
                auto_close=auto_close,
            )
        )
        return handle


class ViserServer(_BackwardsCompatibilityShim if not TYPE_CHECKING else object):
    """:class:`ViserServer` is the main class for working with viser. On
    instantiation, it (a) launches a thread with a web server and (b) provides
    a high-level API for interactive 3D visualization.

    **Core API.** Clients can connect via a web browser, and will be shown two
    components: a 3D scene and a 2D GUI panel. Methods belonging to
    :attr:`ViserServer.scene` can be used to add 3D primitives to the scene.
    Methods belonging to :attr:`ViserServer.gui` can be used to add 2D GUI
    elements.

    **Shared state.** Elements added to the server object, for example via a
    server's :meth:`SceneApi.add_point_cloud` or :meth:`GuiApi.add_button`,
    will have state that's shared and synchronized automatically between all
    connected clients. To show elements that are local to a single client, see
    :attr:`ClientHandle.scene` and :attr:`ClientHandle.gui`.

    Args:
        host: Host to bind server to.
        port: Port to bind server to.
        label: Label shown at the top of the GUI panel.
    """

    # Hide deprecated arguments from docstring and type checkers.
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: str | None = None,
        verbose: bool = True,
        **_deprecated_kwargs,
    ):
        # Create server.
        server = infra.WebsockServer(
            host=host,
            port=port,
            message_class=_messages.Message,
            http_server_root=Path(__file__).absolute().parent / "client" / "build",
            verbose=verbose,
            client_api_version=1,
        )
        self._websock_server = server

        _client_autobuild.ensure_client_is_built()

        self._connection = server
        self._connected_clients: dict[int, ClientHandle] = {}
        self._client_lock = threading.Lock()
        self._client_connect_cb: list[Callable[[ClientHandle], None]] = []
        self._client_disconnect_cb: list[Callable[[ClientHandle], None]] = []

        # For new clients, register and add a handler for camera messages.
        @server.on_client_connect
        def _(conn: infra.WebsockClientConnection) -> None:
            client = ClientHandle(conn, server=self)
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
                    with self._client_lock:
                        self._connected_clients[conn.client_id] = client
                        for cb in self._client_connect_cb:
                            cb(client)

                for camera_cb in client.camera._state.camera_cb:
                    camera_cb(client.camera)

            conn.register_handler(_messages.ViewerCameraMessage, handle_camera_message)

        # Remove clients when they disconnect.
        @server.on_client_disconnect
        def _(conn: infra.WebsockClientConnection) -> None:
            with self._client_lock:
                if conn.client_id not in self._connected_clients:
                    return

                handle = self._connected_clients.pop(conn.client_id)
                for cb in self._client_disconnect_cb:
                    cb(handle)

        # Start the server.
        server.start()

        self.scene: SceneApi = SceneApi(self, thread_executor=server._thread_executor)
        """Handle for interacting with the 3D scene."""

        self.gui: GuiApi = GuiApi(self, thread_executor=server._thread_executor)
        """Handle for interacting with the GUI."""

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

        self._share_tunnel: ViserTunnel | None = None

        # Create share tunnel if requested.
        # This is deprecated: we should use get_share_url() instead.
        share = _deprecated_kwargs.get("share", False)
        if share:
            self.request_share_url()

        self.scene.reset()
        self.gui.reset()
        self.gui.set_panel_label(label)

    def get_host(self) -> str:
        """Returns the host address of the Viser server.

        Returns:
            Host address as string.
        """
        return self._websock_server._host

    def get_port(self) -> int:
        """Returns the port of the Viser server. This could be different from the
        originally requested one.

        Returns:
            Port as integer.
        """
        return self._websock_server._port

    def request_share_url(self, verbose: bool = True) -> str | None:
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

            self._share_tunnel = ViserTunnel(
                "share.viser.studio", self._websock_server._port
            )

            @self._share_tunnel.on_disconnect
            def _() -> None:
                rich.print("[bold](viser)[/bold] Disconnected from share URL")
                self._share_tunnel = None
                self._websock_server.unsafe_send_message(
                    _messages.ShareUrlUpdated(None)
                )

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
                self._websock_server.unsafe_send_message(
                    _messages.ShareUrlUpdated(share_url)
                )
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
        self._websock_server.stop()
        if self._share_tunnel is not None:
            self._share_tunnel.close()

    def get_clients(self) -> dict[int, ClientHandle]:
        """Creates and returns a copy of the mapping from connected client IDs to
        handles.

        Returns:
            Dictionary of clients.
        """
        with self._client_lock:
            return self._connected_clients.copy()

    def on_client_connect(
        self, cb: Callable[[ClientHandle], None]
    ) -> Callable[[ClientHandle], None]:
        """Attach a callback to run for newly connected clients."""
        with self._client_lock:
            clients = self._connected_clients.copy().values()
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

    def flush(self) -> None:
        """Flush the outgoing message buffer. Any buffered messages will immediately be
        sent. (by default they are windowed)"""
        self._websock_server.flush()

    def atomic(self) -> ContextManager[None]:
        """Returns a context where: all outgoing messages are grouped and applied by
        clients atomically.

        This should be treated as a soft constraint that's helpful for things
        like animations, or when we want position and orientation updates to
        happen synchronously.

        Returns:
            Context manager.
        """
        return self._websock_server.atomic()

    def send_file_download(
        self, filename: str, content: bytes, chunk_size: int = 1024 * 1024
    ) -> None:
        """Send a file for a client or clients to download.

        Args:
            filename: Name of the file to send. Used to infer MIME type.
            content: Content of the file.
            chunk_size: Number of bytes to send at a time.
        """
        for client in self.get_clients().values():
            client.send_file_download(filename, content, chunk_size)

    def _start_scene_recording(self) -> RecordHandle:
        """Start recording outgoing messages for playback or
        embedding. Includes only the scene.

        **Work-in-progress.** This API may be changed or removed.
        """
        recorder = self._websock_server.start_recording(
            # Don't record GUI messages. This feels brittle.
            filter=lambda message: "Gui" not in type(message).__name__
        )
        # Insert current scene state.
        for message in self._websock_server._broadcast_buffer.message_from_id.values():
            recorder._insert_message(message)
        return recorder
