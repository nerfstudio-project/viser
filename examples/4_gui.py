"""Examples of basic UI elements that we can create, read from, and write to."""

import time

import numpy as onp

import viser


def main():
    server = viser.ViserServer()

    # Add some common GUI elements: number inputs, sliders, vectors, checkboxes.
    counter = 0

    with server.gui_folder("Read-only"):
        gui_counter = server.add_gui_number(
            "Counter", initial_value=counter
        ).set_disabled(True)
        gui_slider = server.add_gui_slider(
            "Slider", min=0, max=100, step=1, initial_value=counter
        ).set_disabled(True)

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
        with server.gui_folder("Text toggle"):
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
            gui_rgba = server.add_gui_rgba(
                "Color",
                initial_value=(255, 255, 0, 255),
            )

    # Pre-generate a point cloud to send.
    point_positions = onp.random.uniform(low=-1.0, high=1.0, size=(500, 3))
    point_colors = onp.random.randint(0, 256, size=(500, 3))

    while True:
        # We can call `set_value()` to set an input to a particular value.
        gui_counter.set_value(counter)
        gui_slider.set_value(counter % 100)

        # We can call `value()` to read the current value of an input.
        xy = gui_vector2.get_value()
        server.add_frame(
            "/controlled_frame",
            wxyz=(1, 0, 0, 0),
            position=xy + (0,),
        )

        size = gui_vector3.get_value()
        server.add_point_cloud(
            "/controlled_frame/point_cloud",
            position=point_positions * onp.array(size, dtype=onp.float32),
            color=point_colors,
        )

        # We can use `set_disabled()` to enable/disable GUI elements.
        gui_text.set_hidden(gui_checkbox_hide.get_value())
        gui_button.set_hidden(gui_checkbox_hide.get_value())
        gui_rgba.set_disabled(gui_checkbox_disable.get_value())

        counter += 1
        time.sleep(1e-2)


if __name__ == "__main__":
    main()
