"""Theming

Viser is adding support for theming. Work-in-progress.
"""

import time

import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

server = viser.ViserServer()

buttons = (
    TitlebarButton(
        text="Getting Started",
        icon=None,
        href="https://nerf.studio",
    ),
    TitlebarButton(
        text="Github",
        icon="GitHub",
        href="https://github.com/nerfstudio-project/nerfstudio",
    ),
    TitlebarButton(
        text="Documentation",
        icon="Description",
        href="https://docs.nerf.studio",
    ),
)
image = TitlebarImage(
    image_url_light="https://docs.nerf.studio/en/latest/_static/imgs/logo.png",
    image_url_dark="https://docs.nerf.studio/en/latest/_static/imgs/logo-dark.png",
    image_alt="NerfStudio Logo",
    href="https://docs.nerf.studio/",
)

# image = None

titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

server.configure_theme(titlebar_content=titlebar_theme, control_layout="fixed")

spline_positions = ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0))

server.add_catmull_rom_spline(spline_positions)
server.add_cubic_bezier_spline(
    start=(0, 1, 0), end=(1, 0, 0), midA=(0.5, 0.5, 0), midB=(0, 0.5, 0)
)

server.world_axes.visible = True

while True:
    time.sleep(10.0)
