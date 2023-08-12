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


# GUI elements for controllable values.
dark_mode = server.add_gui_checkbox("Dark mode", initial_value=True)
control_layout = server.add_gui_dropdown(
    "Control layout", ("floating", "fixed", "collapsible")
)
brand_color = server.add_gui_rgb("Brand color", (230, 180, 30))
synchronize = server.add_gui_button("Apply theme")


@synchronize.on_click
def synchronize_theme(_) -> None:
    server.configure_theme(
        dark_mode=dark_mode.value,
        titlebar_content=titlebar_theme,
        control_layout=control_layout.value,
        brand_color=brand_color.value,
    )
    server.world_axes.visible = True


synchronize_theme(synchronize)

while True:
    time.sleep(10.0)
