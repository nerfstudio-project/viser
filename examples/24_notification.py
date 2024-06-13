"""Notifications

Examples of adding notifications in Viser."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()

    gui_button = server.gui.add_button("Button")

    @gui_button.on_click
    def _(_) -> None:
        """Reset the scene when the button is clicked."""
        server.gui.add_notification(
            title="Notification",
            body="You have clicked a button!",
            autoClose=True,
            withCloseButton=True,
            loading=False,
        )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
