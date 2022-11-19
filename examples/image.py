import pathlib
import random
import time

import numpy as onp
from PIL import Image

import viser

server = viser.ViserServer()

server.queue(
    viser.ResetSceneMessage(),
    viser.FrameMessage(
        "/main",
        xyzw=(0.0, 0.0, 0.0, 1.0),
        position=(2.0, 2.0, 0.0),
        show_axes=False,
    ),
    viser.ImageMessage.from_image(
        "/main/img",
        Image.open(pathlib.Path("./assets/Cal_logo.png")),
        4.0,
        4.0,
    ),
    viser.FrameMessage(
        "/main/bkgd",
        xyzw=(0.0, 0.0, 0.0, 1.0),
        position=(0.0, 0.0, -1e-5),
        show_axes=False,
    ),
)
while True:
    server.queue(
        viser.ImageMessage.from_array(
            "/main/bkgd/img",
            onp.random.randint(
                0,
                256,
                size=(400, 400, 3),
                dtype=onp.uint8,
            ),
            4.0,
            4.0,
        ),
    )
    time.sleep(0.1)
