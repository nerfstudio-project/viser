"""Notifications

Examples of adding notifications in Viser."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()

    persistent_notif_button = server.gui.add_button(
        "Show persistent notification (default)"
    )
    timed_notif_button = server.gui.add_button("Show timed notification")
    controlled_notif_button = server.gui.add_button("Show controlled notification")
    loading_notif_button = server.gui.add_button("Show loading notification")

    remove_controlled_notif = server.gui.add_button("Remove controlled notification")

    @persistent_notif_button.on_click
    def _(_) -> None:
        """Show persistent notification when the button is clicked."""
        server.gui.add_notification(
            title="Persistent notification",
            body="This can be closed manually and does not disappear on its own!",
            loading=False,
            with_close_button=True,
            auto_close=False,
        )

    @timed_notif_button.on_click
    def _(_) -> None:
        """Show timed notification when the button is clicked."""
        server.gui.add_notification(
            title="Timed notification",
            body="This disappears automatically after 5 seconds!",
            loading=False,
            with_close_button=True,
            auto_close=5000,
        )

    @controlled_notif_button.on_click
    def _(_) -> None:
        """Show controlled notification when the button is clicked."""
        controlled_notif = server.gui.add_notification(
            title="Controlled notification",
            body="This cannot be closed by the user and is controlled in code only!",
            loading=False,
            with_close_button=False,
            auto_close=False,
        )

        @remove_controlled_notif.on_click
        def _(_) -> None:
            """Remove controlled notification."""
            controlled_notif.remove()

    @loading_notif_button.on_click
    def _(_) -> None:
        """Show loading notification when the button is clicked."""
        loading_notif = server.gui.add_notification(
            title="Loading notification",
            body="This indicates that some action is in progress!",
            loading=True,
            with_close_button=True,
            auto_close=False,
        )

        time.sleep(3.0)
        loading_notif.loading = False
        # loading_notif.update(
        #     title="Update notification", body="This notification was updated!"
        # )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
