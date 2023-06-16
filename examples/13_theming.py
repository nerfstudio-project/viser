"""Theming

Viser is adding support for theming. Work-in-progress.
"""

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as onp

import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

server = viser.ViserServer()

buttons = (
    TitlebarButton(
        text="Getting Started",
        icon=None,
        href="https://nerf.studio",
        variant="outlined",
    ),
    TitlebarButton(
        text="Github",
        icon="GitHub",
        href="https://github.com/nerfstudio-project/nerfstudio",
        variant="outlined",
    ),
    TitlebarButton(
        text="Documentation",
        icon="Description",
        href="https://docs.nerf.studio",
        variant="outlined",
    ),
    TitlebarButton(
        text="Viewport Controls",
        icon="Keyboard",
        href="https://docs.nerf.studio",
        variant="outlined",
    ),
)
image = TitlebarImage(
    image_url="https://docs.nerf.studio/en/latest/_static/imgs/logo.png",
    image_alt="NerfStudio Logo",
    href=None,
)

titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

server.configure_theme(
    canvas_background_color=(2, 230, 230),
    titlebar_content=titlebar_theme,
)
server.world_axes.visible = True

while True:
    time.sleep(10.0)
