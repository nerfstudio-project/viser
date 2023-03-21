"""Quick example showing communication on a per-client basis.

We send unique coordinate frames to each connected client.

server.add_*() will add a scene node to all connected clients; the same API is available
via client.add_*().
"""

import time

import imageio.v3 as iio

import viser

server = viser.ViserServer()

# Broadcast a common frame to all connected clients.
# These messages are persistent, so all future clients will also receive them!
server.add_frame("/main", wxyz=(1, 0, 0, 0), position=(0, 0, 0), show_axes=False)
server.add_image(
    "/main/img",
    iio.imread("./assets/Cal_logo.png")[::-1, :],
    4.0,
    4.0,
    format="png",
)

while True:
    # Get all currently connected clients.
    clients = server.get_clients()

    for id, client in clients.items():
        # Match the image rotation of this particular client to face its camera.
        camera = client.get_camera()
        client.add_frame("/main", wxyz=camera.wxyz, position=(0, 0, 0), show_axes=False)

    time.sleep(0.01)
