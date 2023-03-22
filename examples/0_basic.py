"""Basic example: add and remove some coordinate frames."""

import random
import time

import viser

server = viser.ViserServer()

while True:
    server.add_frame(
        "/tree",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    server.add_frame(
        "/tree/branch",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    server.add_frame(
        "/tree/branch/leaf",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    time.sleep(5.0)

    server.remove_scene_node("/tree/branch")
    time.sleep(0.5)
