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
)
image = TitlebarImage(
    image_url_light="https://docs.nerf.studio/en/latest/_static/imgs/logo.png",
    image_url_dark="https://docs.nerf.studio/en/latest/_static/imgs/logo-dark.png",
    image_alt="NerfStudio Logo",
    href="https://docs.nerf.studio/",
)

titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

server.configure_theme(
    titlebar_content=titlebar_theme,
    fixed_sidebar=True,
)
server.world_axes.visible = True

while True:
    time.sleep(10.0)
