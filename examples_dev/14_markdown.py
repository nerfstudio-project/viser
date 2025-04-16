"""Markdown demonstration

Viser GUI has MDX 2 support.
"""

import time
from pathlib import Path

import viser

server = viser.ViserServer()
server.scene.world_axes.visible = True

markdown_counter = server.gui.add_markdown("Counter: 0")

here = Path(__file__).absolute().parent

button = server.gui.add_button("Remove blurb")
checkbox = server.gui.add_checkbox("Visibility", initial_value=True)

markdown_source = (here / "./assets/mdx_example.mdx").read_text()
markdown_blurb = server.gui.add_markdown(
    content=markdown_source,
    image_root=here,
)


@button.on_click
def _(_):
    markdown_blurb.remove()


@checkbox.on_update
def _(_):
    markdown_blurb.visible = checkbox.value


counter = 0
while True:
    markdown_counter.content = f"Counter: {counter}"
    counter += 1
    time.sleep(0.1)
