"""Depth compositing

In this example, we show how to use a background image with depth compositing. This can
be useful when we want a 2D image to occlude 3D geometry, such as for NeRF rendering.
"""

import time

import numpy as np
import trimesh
import trimesh.creation

import viser

server = viser.ViserServer()


img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
depth = np.ones((1000, 1000, 1), dtype=np.float32)

# Make a square middle portal.
depth[250:750, 250:750, :] = 10.0
img[250:750, 250:750, :] = 255

mesh = trimesh.creation.box((0.5, 0.5, 0.5))
server.scene.add_mesh_trimesh(
    name="/cube",
    mesh=mesh,
    position=(0, 0, 0.0),
)
server.scene.set_background_image(img, depth=depth)


while True:
    time.sleep(1.0)
