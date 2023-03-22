"""Asynchronous usage of GUI elements: we can attach callbacks that are called as soon
as we get updates."""

import time

import numpy as onp
from typing_extensions import assert_never

import viser

server = viser.ViserServer()

with server.gui_folder("Control"):
    gui_show = server.add_gui_checkbox("Show Frame", initial_value=True)
    gui_axis = server.add_gui_select("Axis", options=["x", "y", "z"])
    gui_location = server.add_gui_slider(
        "Location", min=-5.0, max=5.0, step=0.05, initial_value=0.0
    )
    gui_num_points = server.add_gui_slider(
        "# Points", min=1000, max=200_000, step=1000, initial_value=10_000
    )

with server.gui_folder("Reset"):
    gui_button = server.add_gui_button("Reset")

# We can read from each of these via the `.value()` function.
# Let's define some functions that read the values, then use them to update a
# frame/point cloud pair in the scene..


def draw_frame():
    axis = gui_axis.value()
    if axis == "x":
        pos = (gui_location.value(), 0.0, 0.0)
    elif axis == "y":
        pos = (0.0, gui_location.value(), 0.0)
    elif axis == "z":
        pos = (0.0, 0.0, gui_location.value())
    else:
        assert_never(axis)

    server.add_frame(
        "/frame",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=pos,
        show_axes=gui_show.value(),
        scale=5.0,
    )


def draw_points():
    num_points = gui_num_points.value()
    server.add_point_cloud(
        "/frame/point_cloud",
        position=onp.random.normal(size=(num_points, 3)),
        color=onp.random.randint(0, 256, size=(num_points, 3)),
    )


# We can (optionally) also attach callbacks!
# Here, we update the point clouds + frames whenever any of the GUI items are updated.


@gui_show.on_update
def show_cb(value: bool) -> None:
    print("Got new show:", value)
    draw_frame()


@gui_axis.on_update
def axis_cb(value: str) -> None:
    print("Got new axis:", value)
    draw_frame()


@gui_location.on_update
def location_cb(value: float) -> None:
    print("Got new location:", value)
    draw_frame()


@gui_num_points.on_update
def num_points_cb(value: float) -> None:
    print("Got new point count:", value)
    draw_points()


@gui_button.on_update
def reset(value: bool):
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
    time.sleep(10.0)
