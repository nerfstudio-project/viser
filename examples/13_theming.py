# mypy: disable-error-code="arg-type"
#
# Waiting on PEP 675 support in mypy. https://github.com/python/mypy/issues/12554
"""Theming

Viser includes support for light theming.
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
titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

# server.add_gui_markdown(
#     "Viser includes support for light theming via the `.configure_theme()` method."
# )

# GUI elements for controllable values.
for i in range(30):
    control_layout = server.add_gui_dropdown(
        "Control layout", ("floating", "fixed", "collapsible")
    )
titlebar = server.add_gui_checkbox("Titlebar", initial_value=True)
titlebar.visible = False
dark_mode = server.add_gui_checkbox("Dark mode", initial_value=True)
dark_mode.visible = False
brand_color = server.add_gui_rgb("Brand color", (230, 180, 30))
brand_color.visible = False
synchronize = server.add_gui_button("Apply theme", icon=viser.Icon.CHECK)


def synchronize_theme() -> None:
    server.configure_theme(
        dark_mode=dark_mode.value,
        titlebar_content=titlebar_theme if titlebar.value else None,
        control_layout=control_layout.value,
        brand_color=brand_color.value,
    )
    server.world_axes.visible = True


synchronize.on_click(lambda _: synchronize_theme())
synchronize_theme()

while True:
    time.sleep(10.0)
