"""Markdown Demonstration

Viser GUI has MDX 2 support.
"""

import time
from pathlib import Path

import viser

server = viser.ViserServer()
server.world_axes.visible = True


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    here = Path(__file__).absolute().parent
    markdown_source = (here / "./assets/mdx_example.mdx").read_text()
    markdown = client.add_gui_markdown(markdown=markdown_source, image_root=here)

    button = client.add_gui_button("Remove Markdown")
    checkbox = client.add_gui_checkbox("Visibility", initial_value=True)

    @button.on_click
    def _(_):
        markdown.remove()

    @checkbox.on_update
    def _(_):
        markdown.visible = checkbox.value


while True:
    time.sleep(10.0)
