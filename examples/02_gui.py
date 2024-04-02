# mypy: disable-error-code="assignment"
#
# Asymmetric properties are supported in Pyright, but not yet in mypy.
# - https://github.com/python/mypy/issues/3004
# - https://github.com/python/mypy/pull/11643
"""GUI basics

Examples of basic GUI elements that we can create, read from, and write to."""

import time

import numpy as onp
import viser

from PIL import Image
import plotly.express as px

def main() -> None:
    server = viser.ViserServer(port=8008)

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
            gui_multi_slider = server.add_gui_multi_slider(
                "Multi slider",
                min=0,
                max=100,
                step=1,
                initial_value=(0, 30, 100),
                marks=((0, "0"), (50, "5"), (70, "7"), 99),
            )
            gui_slider_positions = server.add_gui_slider(
                "# sliders",
                min=0,
                max=10,
                step=1,
                initial_value=3,
                marks=((0, "0"), (5, "5"), (7, "7"), 10),
            )
            gui_upload_button = server.add_gui_upload_button(
                "Upload", icon=viser.Icon.UPLOAD
            )

    @gui_upload_button.on_upload
    def _(_) -> None:
        """Callback for when a file is uploaded."""
        file = gui_upload_button.value
        print(file.name, len(file.content), "bytes")

    # Pre-generate a point cloud to send.
    point_positions = onp.random.uniform(low=-1.0, high=1.0, size=(5000, 3))
    color_coeffs = onp.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))

    with server.add_gui_folder("Plots"):
        cal_img_path = 'examples/assets/Cal_logo.png'
        img = Image.open(cal_img_path)
        fig = px.imshow(img)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)

        server.add_gui_plotly(
            label="Image plot!",
            figure=fig,
            aspect_ratio=1.0,
        )

        x_data = onp.linspace(0, 6*onp.pi, 50)
        y_data = onp.sin(x_data) * 10
        x_data, y_data = list(x_data), list(y_data)

        fig = px.line(x=x_data, y=y_data, labels={'x': 'x', 'y': 'sin(x)'})
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)

        server.add_gui_plotly(
            label="Line plot!",
            figure=fig,
            aspect_ratio=1.0,
        )

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
            point_shape="circle",
        )

        # We can use `.visible` and `.disabled` to toggle GUI elements.
        gui_text.visible = not gui_checkbox_hide.value
        gui_button.visible = not gui_checkbox_hide.value
        gui_rgb.disabled = gui_checkbox_disable.value
        gui_button.disabled = gui_checkbox_disable.value
        gui_upload_button.disabled = gui_checkbox_disable.value

        # Update the number of handles in the multi-slider.
        if gui_slider_positions.value != len(gui_multi_slider.value):
            gui_multi_slider.value = onp.linspace(
                0, 100, gui_slider_positions.value, dtype=onp.int64
            )

        counter += 1
        time.sleep(0.01)


if __name__ == "__main__":
    main()
