"""Images

Example for sending images to the viewer.

We can send backgrond images to display behind the viewer (useful for visualizing
NeRFs), or images to render as 3D textures.
"""

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as onp

import viser

from viser.theme import TitlebarConfig, TitlebarButton, TitlebarImage

server = viser.ViserServer()

# Add a background image.
server.set_background_image(
    iio.imread(Path(__file__).parent / "assets/Cal_logo.png"),
    format="png",
)

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

# Add main image.
server.add_image(
    "/img",
    iio.imread(Path(__file__).parent / "assets/Cal_logo.png"),
    4.0,
    4.0,
    format="png",
    wxyz=(1.0, 0.0, 0.0, 0.0),
    position=(2.0, 2.0, 0.0),
)
while True:
    server.add_image(
        "/noise",
        onp.random.randint(
            0,
            256,
            size=(400, 400, 3),
            dtype=onp.uint8,
        ),
        4.0,
        4.0,
        format="jpeg",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(2.0, 2.0, -1e-2),
    )
    time.sleep(0.2)
