"""Notifications

Examples of adding notifications in Viser."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()

    gui_button = server.gui.add_button("Button")
    clear_button = server.gui.add_button("Clear Notifications")

    @gui_button.on_click
    def _(_) -> None:
        """Reset the scene when the button is clicked."""
        notif = server.gui.add_notification(
            title="Notification",
            body="You have clicked a button!",
            withCloseButton=True,
            loading=False,
            autoClose=2000,
        )
        notif.show()

    @clear_button.on_click
    def _(_) -> None:
        """Clear all open notifcations."""
        server.gui.clear_notification()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
