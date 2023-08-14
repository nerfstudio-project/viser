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
import trimesh 

server = viser.ViserServer()


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    img = onp.random.randint(0,255, size=(1000, 1000, 3), dtype=onp.uint8)
    depth = onp.ones((1000,1000,1), dtype=onp.float32)
    #make a middle square portal
    depth[250:750,250:750,:] = 10.0
    img[250:750,250:750,:] = 255
    mesh = trimesh.creation.box((0.5, 0.5, 0.5))
    handle = server.add_mesh_trimesh(
                name=f"/cube",
                mesh=mesh,
                position=(0,0, 0.0),
            )
    client.set_popup_image(img,depth)
while True:
    time.sleep(1.0)
