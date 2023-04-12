# mypy: disable-error-code="arg-type"
#
# assert_never() on the `axis` variable is waiting on PEP 675 support in mypy.
# https://github.com/python/mypy/issues/12554
#
# In the meantime, it works great in Pyright/Pylance, which is more important.

"""Asynchronous usage of GUI elements: we can attach callbacks that are called as soon
as we get updates."""

import time

import numpy as onp
from typing_extensions import assert_never

import viser


def main():
    server = viser.ViserServer()

    with server.gui_folder("Control"):
        gui_show = server.add_gui_checkbox("Show Frame", initial_value=True)
        gui_axis = server.add_gui_select("Axis", options=["x", "y", "z"])
        gui_include_z = server.add_gui_checkbox("Z in dropdown", initial_value=True)

        @gui_include_z.on_update
        def _(_) -> None:
            gui_axis.set_options(
                ["x", "y", "z"] if gui_include_z.get_value() else ["x", "y"]
            )

        with server.gui_folder("Sliders"):
            gui_location = server.add_gui_slider(
                "Location", min=-5.0, max=5.0, step=0.05, initial_value=0.0
            )
            gui_num_points = server.add_gui_slider(
                "# Points", min=1000, max=200_000, step=1000, initial_value=10_000
            )

    with server.gui_folder("Reset"):
        gui_button = server.add_gui_button("Reset")

    def draw_frame() -> None:
        axis = gui_axis.get_value()
        if axis == "x":
            pos = (gui_location.get_value(), 0.0, 0.0)
        elif axis == "y":
            pos = (0.0, gui_location.get_value(), 0.0)
        elif axis == "z":
            pos = (0.0, 0.0, gui_location.get_value())
        else:
            assert_never(axis)

        server.add_frame(
            "/frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=pos,
            show_axes=gui_show.get_value(),
            axes_length=5.0,
        )

    def draw_points() -> None:
        num_points = gui_num_points.get_value()
        server.add_point_cloud(
            "/frame/point_cloud",
            position=onp.random.normal(size=(num_points, 3)),
            color=onp.random.randint(0, 256, size=(num_points, 3)),
        )

    # We can (optionally) also attach callbacks!
    # Here, we update the point clouds + frames whenever any of the GUI items are updated.
    gui_show.on_update(lambda _: draw_frame())
    gui_axis.on_update(lambda _: draw_frame())
    gui_location.on_update(lambda _: draw_frame())
    gui_num_points.on_update(lambda _: draw_points())

    @gui_button.on_update
    def _(_: viser.GuiHandle[bool]) -> None:
        """Reset the scene when the reset button is clicked."""
        gui_show.set_value(True)
        gui_location.set_value(0.0)
        gui_axis.set_value("x")
        gui_num_points.set_value(10_000)

        draw_frame()
        draw_points()

    # Finally, let's add the initial frame + point cloud and just loop infinitely. :)
    draw_frame()
    draw_points()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
