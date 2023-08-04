"""Modal basics

Examples of using modals in Viser."""

import time

import numpy as onp

import viser


def main():
    server = viser.ViserServer()

    modal = server.add_modal(
        "Modal",
        initial_value="Modals support all control panel elements!",
    )

    default_text_entry = "Tell me you name!"

    with server.gui_destination("MODAL"):

        name_text = server.add_gui_text(
            "Your Name",
            initial_value=default_text_entry,
        )

        modal_button = server.add_gui_button(
            "Enter your name",
            disabled=True,
            visible=True
        )

        global control_panel_activated
        control_panel_activated = False
        global control_panel_text
        control_panel_text = None
        @modal_button.on_click
        def _(_: viser.GuiButtonHandle) -> None:
            modal.visible = False

            global control_panel_activated
            if not control_panel_activated:
                control_panel_activated = True
                with server.gui_destination("CONTROL_PANEL"):
                    global control_panel_text
                    control_panel_text = server.add_gui_text(
                        "Greeting:",
                        initial_value="",
                        disabled=True
                    )
                    control_panel_button = server.add_gui_button(
                        "Re-open modal",
                        visible=True
                    )
                    @control_panel_button.on_click
                    def _(_: viser.GuiButtonHandle) -> None:
                        modal.visible = True

            control_panel_text.value = f"Hello, {name_text.value}!"

        with server.gui_folder("[GUI Examples]"):
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

            gui_rgb = server.add_gui_rgb(
                "Color",
                initial_value=(255, 255, 0),
            )


    counter = 0
    while True:
        gui_counter.value = counter
        gui_slider.value = counter % 100

        modal_button.disabled = name_text.value == default_text_entry
        if (name_text.value != default_text_entry):
            modal_button.name = "Close Modal"


        counter += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()
