# mypy: disable-error-code="assignment"
#
# Asymmetric properties are supported in Pyright, but not yet in mypy.
# - https://github.com/python/mypy/issues/3004
# - https://github.com/python/mypy/pull/11643
"""3D GUI Elements

`add_3d_gui_container()` allows standard GUI elements to be incorporated directly into a
3D scene. In this example, we click on coordinate frames to show actions that can be
performed on them.
"""

import time
from typing import Optional

import numpy as onp

import viser
import viser.transforms as tf
import trimesh 

server = viser.ViserServer()
num_frames = 20


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    img = onp.random.randint(0,255, size=(100, 100, 3), dtype=onp.uint8)
    depth = onp.eye(100, dtype=onp.float32)[...,None]+.5
    # depth = onp.random.rand(100,100,1).astype(onp.float32)
    print(depth.shape)
    mesh = trimesh.creation.box((0.5, 0.5, 0.5))
    handle = server.add_mesh_trimesh(
                name=f"/cube",
                mesh=mesh,
                position=(0,0, 0.0),
            )
    client.set_popup_image(img,depth)
while True:
    time.sleep(1.0)
