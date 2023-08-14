"""Modal basics

Examples of using modals in Viser."""

import time

import viser


def main():
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        with client.add_gui_modal("Modal example"):
            client.add_gui_markdown(
                markdown=(
                    "**The slider below determines how many modals will appear...**"
                )
            )

            gui_slider = client.add_gui_slider(
                "Slider",
                min=1,
                max=10,
                step=1,
                initial_value=1,
            )

            modal_button = client.add_gui_button("Show more modals")

            @modal_button.on_click
            def _(_: viser.GuiButtonHandle) -> None:
                for i in range(gui_slider.value):
                    with client.add_gui_modal(f"Modal #{i}"):
                        client.add_gui_markdown("This is a modal!")

    while True:
        time.sleep(0.15)


if __name__ == "__main__":
    main()
