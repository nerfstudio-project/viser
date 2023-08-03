"""Coordinate frames

In this basic example, we visualize a set of coordinate frames.

Naming for all scene nodes are hierarchical; /tree/branch, for example, is defined
relative to /tree.
"""

import random
import time

import viser

server = viser.ViserServer(share=True)

while True:
    # Add some coordinate frames to the scene. These will be visualized in the viewer.
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
    leaf = server.add_frame(
        "/tree/branch/leaf",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    time.sleep(5.0)

    # Remove the leaf node from the scene.
    leaf.remove()
    time.sleep(0.5)
