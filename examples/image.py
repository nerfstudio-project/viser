import pathlib
import random
import time

import imageio.v3 as iio
import numpy as onp

import viser

server = viser.ViserServer()

server.queue(
    viser.ResetSceneMessage(),
    viser.FrameMessage(
        "/main",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(2.0, 2.0, 0.0),
        show_axes=False,
    ),
    viser.BackgroundImageMessage.encode(
        iio.imread("./assets/Cal_logo.png"), format="png"
    ),
    viser.ImageMessage.encode(
        "/main/img",
        iio.imread("./assets/Cal_logo.png"),
        4.0,
        4.0,
        format="png",
    ),
    viser.FrameMessage(
        "/main/bkgd",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, -1e-2),
        show_axes=False,
    ),
)
while True:
    server.queue(
        viser.ImageMessage.encode(
            "/main/bkgd/img",
            onp.random.randint(
                0,
                256,
                size=(400, 400, 3),
                dtype=onp.uint8,
            ),
            4.0,
            4.0,
            format="jpeg",
        ),
    )
    time.sleep(0.1)
