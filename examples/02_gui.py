"""GUI basics

Examples of basic GUI elements that we can create, read from, and write to."""

import time

import numpy as onp

import viser


def main():
    server = viser.ViserServer()

    # Add some common GUI elements: number inputs, sliders, vectors, checkboxes.
    with server.add_gui_folder("Read-only"):
        gui_counter = server.add_gui_number(
            "Counter",
            initial_value=0,
            disabled=True,
        )

        gui_slider = server.add_gui_slider(
            "Slider",
            min=0,
            max=100,
            step=1,
            initial_value=0,
            disabled=True,
        )

    with server.add_gui_folder("Editable"):
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
        with server.add_gui_folder("Text toggle"):
            gui_checkbox_hide = server.add_gui_checkbox(
                "Hide",
                initial_value=False,
            )
            gui_text = server.add_gui_text(
                "Text",
                initial_value="Hello world",
            )
            gui_button = server.add_gui_button("Button")
            gui_checkbox_disable = server.add_gui_checkbox(
                "Disable",
                initial_value=False,
            )
            gui_rgb = server.add_gui_rgb(
                "Color",
                initial_value=(255, 255, 0),
            )

    # Pre-generate a point cloud to send.
    point_positions = onp.random.uniform(low=-1.0, high=1.0, size=(5000, 3))
    color_coeffs = onp.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))

    counter = 0
    while True:
        # We can set the value of an input to a particular value. Changes are
        # automatically reflected in connected clients.
        gui_counter.value = counter
        gui_slider.value = counter % 100

        # We can set the position of a scene node with `.position`, and read the value
        # of a gui element with `.value`. Changes are automatically reflected in
        # connected clients.
        server.add_point_cloud(
            "/point_cloud",
            points=point_positions * onp.array(gui_vector3.value, dtype=onp.float32),
            colors=(
                onp.tile(gui_rgb.value, point_positions.shape[0]).reshape((-1, 3))
                * color_coeffs[:, None]
            ).astype(onp.uint8),
            position=gui_vector2.value + (0,),
        )

        # We can use `.visible` and `.disabled` to toggle GUI elements.
        gui_text.visible = not gui_checkbox_hide.value
        gui_button.visible = not gui_checkbox_hide.value
        gui_rgb.disabled = gui_checkbox_disable.value

        counter += 1
        time.sleep(0.01)


if __name__ == "__main__":
    main()
