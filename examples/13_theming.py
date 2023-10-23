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
server.world_axes.visible = True

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
    image_url_light="https://docs.nerf.studio/_static/imgs/logo.png",
    image_url_dark="https://docs.nerf.studio/_static/imgs/logo-dark.png",
    image_alt="NerfStudio Logo",
    href="https://docs.nerf.studio/",
)
titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

server.add_gui_markdown(
    "Viser includes support for light theming via the `.configure_theme()` method."
)

gui_theme_code = server.add_gui_markdown("no theme applied yet")

# GUI elements for controllable values.
titlebar = server.add_gui_checkbox("Titlebar", initial_value=True)
dark_mode = server.add_gui_checkbox("Dark mode", initial_value=True)
show_logo = server.add_gui_checkbox("Show logo", initial_value=True)
brand_color = server.add_gui_rgb("Brand color", (230, 180, 30))
control_layout = server.add_gui_dropdown(
    "Control layout", ("floating", "fixed", "collapsible")
)
synchronize = server.add_gui_button("Apply theme", icon=viser.Icon.CHECK)


def synchronize_theme() -> None:
    server.configure_theme(
        titlebar_content=titlebar_theme if titlebar.value else None,
        control_layout=control_layout.value,
        dark_mode=dark_mode.value,
        show_logo=show_logo.value,
        brand_color=brand_color.value,
    )
    gui_theme_code.content = f"""
        ### Current applied theme
        ```
        server.configure_theme(
            titlebar_content={"titlebar_content" if titlebar.value else None},
            control_layout="{control_layout.value}",
            dark_mode={dark_mode.value},
            show_logo={show_logo.value},
            brand_color={brand_color.value},
        )
        ```
    """


synchronize.on_click(lambda _: synchronize_theme())
synchronize_theme()

while True:
    time.sleep(10.0)
