# mypy: disable-error-code="assignment"
#
# Asymmetric properties are supported in Pyright, but not yet in mypy.
# - https://github.com/python/mypy/issues/3004
# - https://github.com/python/mypy/pull/11643
"""Popup image example.
In this example, we show how to use the popup image feature.
It shows a 2.5d RGBD image that tracks the camera around and lays over other geometry in the scene.
"""

import time

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
