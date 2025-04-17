"""Coordinate frames

In this basic example, we visualize a set of coordinate frames.

Naming for all scene nodes are hierarchical; /tree/branch, for example, is defined
relative to /tree.
"""

import random
import time

import numpy as np
import viser
import viser.transforms as tf

server = viser.ViserServer()

# Add some coordinate frames to the scene. These will be visualized in the viewer.
frame = server.scene.add_frame(
    "/tree",
    wxyz=(1.0, 0.0, 0.0, 0.0),
    position=(1.0, 0.0, 0.0),
)

@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    @frame.on_click
    def _(_):
        print("Moving camera to frame")
        # Get frame's position and orientation
        target_position = frame.position
        target_wxyz = frame.wxyz

        # Atomically update camera position and orientation
        with client.atomic():
            client.camera.position = target_position
            client.camera.wxyz = target_wxyz

        # Set look_at *after* moving the camera
        # Match the example: set look_at directly to the frame position
        client.camera.look_at = target_position

while True:
    time.sleep(4)
