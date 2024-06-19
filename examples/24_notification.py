"""Notifications

Examples of adding notifications in Viser."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()

    persistent_notif = server.gui.add_button("Show persistent notification (default)")
    timed_notif = server.gui.add_button("Show timed notification")
    controlled_notif = server.gui.add_button("Show controlled notification")
    loading_notif = server.gui.add_button("Show loading notification")
    clear_all_notif = server.gui.add_button("Clear Notifications")

    @persistent_notif.on_click
    def _(_) -> None:
        """Reset the scene when the button is clicked."""
        server.gui.add_notification(
            title="Persistent notification",
            body="This can be closed manually and does not disappear on its own!",
            loading=False,
            type="persistent",
        )

    @timed_notif.on_click
    def _(_) -> None:
        server.gui.add_notification(
            title="Timed notification",
            body="This disappears automatically after 5 seconds!",
            loading=False,
            type="timed"
        )

    @controlled_notif.on_click
    def _(_) -> None:
        server.gui.add_notification(
            title="Controlled notification",
            body="This cannot be closed by the user and is controlled in code only!",
            loading=False,
            type="controlled"
        )

    @loading_notif.on_click
    def _(_) -> None:
        server.gui.add_notification(
            title="Loading notification",
            body="This indicates that some action is in progress!",
            loading=True,
            type="persistent"
        )

    @clear_all_notif.on_click
    def _(_) -> None:
        """Clear all open notifcations."""
        server.gui.clear_notification()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
