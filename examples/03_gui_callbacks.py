# mypy: disable-error-code="arg-type"
#
# assert_never() on the `axis` variable works in Pyright, but is waiting on PEP 675
# support in mypy. https://github.com/python/mypy/issues/12554
"""GUI callbacks

Asynchronous usage of GUI elements: we can attach callbacks that are called as soon as
we get updates."""

import time

import numpy as onp
from typing_extensions import assert_never

import viser


def main() -> None:
    server = viser.ViserServer()

    with server.gui_folder("Control"):
        gui_show = server.add_gui_checkbox("Show Frame", initial_value=True)
        gui_axis = server.add_gui_select("Axis", options=["x", "y", "z"])
        gui_include_z = server.add_gui_checkbox("Z in dropdown", initial_value=True)

        @gui_include_z.on_update
        def _(_) -> None:
            gui_axis.options = ["x", "y", "z"] if gui_include_z.value else ["x", "y"]

        with server.gui_folder("Sliders"):
            gui_location = server.add_gui_slider(
                "Location", min=-5.0, max=5.0, step=0.05, initial_value=0.0
            )
            gui_num_points = server.add_gui_slider(
                "# Points", min=1000, max=200_000, step=1000, initial_value=10_000
            )

    gui_reset_scene = server.add_gui_button("Reset Scene")

    def draw_frame() -> None:
        axis = gui_axis.value
        if axis == "x":
            pos = (gui_location.value, 0.0, 0.0)
        elif axis == "y":
            pos = (0.0, gui_location.value, 0.0)
        elif axis == "z":
            pos = (0.0, 0.0, gui_location.value)
        else:
            assert_never(axis)

        server.add_frame(
            "/frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=pos,
            show_axes=gui_show.value,
            axes_length=5.0,
        )

    def draw_points() -> None:
        num_points = gui_num_points.value
        server.add_point_cloud(
            "/frame/point_cloud",
            points=onp.random.normal(size=(num_points, 3)),
            colors=onp.random.randint(0, 256, size=(num_points, 3)),
        )

    # We can (optionally) also attach callbacks!
    # Here, we update the point clouds + frames whenever any of the GUI items are updated.
    gui_show.on_update(lambda _: draw_frame())
    gui_axis.on_update(lambda _: draw_frame())
    gui_location.on_update(lambda _: draw_frame())
    gui_num_points.on_update(lambda _: draw_points())

    @gui_reset_scene.on_click
    def _(_: viser.GuiButtonHandle) -> None:
        """Reset the scene when the reset button is clicked."""
        gui_show.value = True
        gui_location.value = 0.0
        gui_axis.value = "x"
        gui_num_points.value = 10_000

        draw_frame()
        draw_points()

    # Finally, let's add the initial frame + point cloud and just loop infinitely. :)
    draw_frame()
    draw_points()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
