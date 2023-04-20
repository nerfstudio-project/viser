"""Example showing communication on a per-client basis.

We send unique coordinate frames to each connected client.

server.add_*() will add a scene node to all connected clients; the same API is available
via client.add_*().
"""

import time
from pathlib import Path

import imageio.v3 as iio

import viser

server = viser.ViserServer()

# Broadcast a common frame to all connected clients.
# These messages are persistent, so all future clients will also receive them!
server.add_frame("/main", wxyz=(1, 0, 0, 0), position=(0, 0, 0), show_axes=False)
server.add_image(
    "/main/img",
    iio.imread(Path(__file__).parent / "assets/Cal_logo.png")[::-1, :],
    4.0,
    4.0,
    format="png",
)

while True:
    # Get all currently connected clients.
    clients = server.get_clients()

    for id, client in clients.items():
        # Match the image rotation of this particular client to face its camera.
        camera = client.camera
        client.add_frame("/main", wxyz=camera.wxyz, position=(0, 0, 0), show_axes=False)

        # Kind of fun: send our own camera to all of the other clients. This lets each
        # connected client see the other clients.
        for other in clients.values():
            if client.client_id == other.client_id:
                continue
            camera = client.camera
            other.add_frame(
                f"/client_{client.client_id}",
                wxyz=camera.wxyz,
                position=camera.position,
                axes_length=0.1,
                axes_radius=0.005,
            )
            other.add_camera_frustum(
                f"/client_{client.client_id}/frustum",
                fov=camera.fov,
                aspect=camera.aspect,
            )

    time.sleep(0.01)
