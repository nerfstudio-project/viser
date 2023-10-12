"""Get Renders

Example for getting renders from a client's viewport to the Python API."""

import time

import imageio.v3 as iio
import numpy as onp

import viser


def main():
    server = viser.ViserServer()

    button = server.add_gui_button("Render a GIF")

    @button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        client.reset_scene()

        images = []

        for i in range(20):
            positions = onp.random.normal(size=(30, 3)) * 3.0
            client.add_spline_catmull_rom(
                f"/catmull_{i}",
                positions,
                tension=0.5,
                line_width=3.0,
                color=onp.random.uniform(size=3),
            )
            images.append(client.camera.get_render(height=720, width=1280))

        print("Writing GIF...")
        iio.imwrite("saved.gif", images)
        print("Wrote GIF!")

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
