"""Markdown Demonstration

Viser GUI has MDX 2 support (WIP)
"""

import time
from pathlib import Path

import imageio.v3 as iio
import viser

server = viser.ViserServer()
server.world_axes.visible = True

with open("./assets/mdx_example.mdx", "r") as mkdn:
    markdown = server.add_gui_markdown(
        markdown=mkdn.read(),
        images={"cal_logo": iio.imread(Path(__file__).parent / "assets/Cal_logo.png")},
    )

button = server.add_gui_button("Remove Markdown")


@button.on_click
def _(_):
    markdown.remove()


while True:
    time.sleep(10.0)
