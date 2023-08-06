"""Markdown Demonstration

Viser GUI has MDX 2 support (WIP)
"""

import time
from pathlib import Path

import viser

server = viser.ViserServer()
server.world_axes.visible = True


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    with open("./assets/mdx_example.mdx", "r") as mkdn:
        markdown = client.add_gui_markdown(
            markdown=mkdn.read(), image_root=Path(__file__).parent
        )

    button = client.add_gui_button("Remove Markdown")

    @button.on_click
    def _(_):
        markdown.remove()


while True:
    time.sleep(10.0)
