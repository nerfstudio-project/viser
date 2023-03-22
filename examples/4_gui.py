import time

import numpy as onp
from typing_extensions import assert_never

import viser

server = viser.ViserServer()
server.reset_scene()

# Add some common GUI elements: a checkbox, dropdown, and slider.

gui_show = server.add_gui_checkbox("Show Frame", initial_value=True)
gui_axis = server.add_gui_select("Axis", options=["x", "y", "z"])
gui_location = server.add_gui_slider(
    "Location", min=-5.0, max=5.0, step=0.05, initial_value=0.0
)
gui_num_points = server.add_gui_slider(
    "# Points", min=1000, max=200_000, step=1000, initial_value=10_000
)


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
        position_f32=onp.random.normal(size=(num_points, 3)).astype(onp.float32),
        color_uint8=onp.random.randint(0, 256, size=(num_points, 3)).astype(onp.uint8),
    )


draw_frame()
draw_points()
while True:
    draw_frame()
    draw_points()
    time.sleep(0.1)
