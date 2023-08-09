"""Modal basics

Examples of using modals in Viser."""

import time

import numpy as onp

import viser


def main():
    server = viser.ViserServer()

    with server.add_gui_modal(""):

        server.add_gui_markdown(markdown="## Hello!")
        server.add_gui_markdown(markdown="#### The slider below determines how many modals will appear...")

        gui_slider = server.add_gui_slider(
            "Slider",
            min=0,
            max=5,
            step=1,
            initial_value=0,
            disabled=True,
        )

        modal_button = server.add_gui_button("Show modals")
        @modal_button.on_click
        def _(_: viser.GuiButtonHandle) -> None:
            ctr = gui_slider.value
            for i in range(ctr):
                    with server.add_gui_modal(f""):
                        server.add_gui_markdown(markdown=f"## This is modal #{i}")
                        server.add_gui_rgb(
                            "Color",
                            initial_value=(255, 255, 0),
                        )


    counter = 0
    while True:
        gui_slider.value = counter % 5

        counter += 1
        time.sleep(0.15)


if __name__ == "__main__":
    main()
