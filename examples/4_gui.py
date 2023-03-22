"""Examples of basic UI elements that we can create, read from, and write to."""

import time

import numpy as onp
from typing_extensions import assert_never

import viser

server = viser.ViserServer()
server.reset_scene()

# Add some common GUI elements: number inputs, sliders, vectors, checkboxes.
counter = 0

with server.gui_folder("Read-only"):
    gui_counter = server.add_gui_number(
        "Counter",
        initial_value=counter,
        disabled=True,
    )
    gui_slider = server.add_gui_slider(
        "Slider",
        min=0,
        max=100,
        step=1,
        initial_value=counter,
        disabled=True,
    )

with server.gui_folder("Editable"):
    gui_vector2 = server.add_gui_vector2(
        "Position",
        initial_value=(0.0, 0.0),
        step=0.1,
    )
    gui_vector3 = server.add_gui_vector3(
        "Size",
        initial_value=(1.0, 1.0, 1.0),
        step=0.25,
        lock=True,
    )
    gui_text = server.add_gui_text(
        "Text",
        initial_value="Hello world",
    )
    gui_checkbox = server.add_gui_checkbox(
        "Disable text",
        initial_value=False,
    )

# Pre-generate a point cloud to send.
point_positions = onp.random.uniform(low=-1.0, high=1.0, size=(500, 3))
point_colors = onp.random.randint(0, 256, size=(500, 3))

while True:
    # We can call `set_value()` to set an input to a particular value.
    gui_counter.set_value(counter)
    gui_slider.set_value(counter % 100)

    # We can call `value()` to read the current value of an input.
    xy = gui_vector2.value()
    server.add_frame(
        "/controlled_frame",
        wxyz=(1, 0, 0, 0),
        position=xy + (0,),
    )

    size = gui_vector3.value()
    server.add_point_cloud(
        "/controlled_frame/point_cloud",
        position=point_positions * onp.array(size, dtype=onp.float32),
        color=point_colors,
    )

    # We can use `set_disabled()` to enable/disable GUI elements.
    gui_text.set_disabled(gui_checkbox.value())

    counter += 1
    time.sleep(0.1)
