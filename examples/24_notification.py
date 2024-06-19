"""Notifications

Examples of adding notifications in Viser."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()

    persistent_notif_button = server.gui.add_button("Show persistent notification (default)")
    timed_notif_button = server.gui.add_button("Show timed notification")
    controlled_notif_button = server.gui.add_button("Show controlled notification")
    loading_notif_button = server.gui.add_button("Show loading notification")

    close_controlled_notif = server.gui.add_button("Clear controlled notification")
    clear_all_notif = server.gui.add_button("Clear all notifications")

    @persistent_notif_button.on_click
    def _(_) -> None:
        """Show persistent notification when the button is clicked."""
        server.gui.add_notification(
            title="Persistent notification",
            body="This can be closed manually and does not disappear on its own!",
            loading=False,
            type="persistent",
        )

    @timed_notif_button.on_click
    def _(_) -> None:
        """Show timed notification when the button is clicked."""
        server.gui.add_notification(
            title="Timed notification",
            body="This disappears automatically after 5 seconds!",
            loading=False,
            type="timed"
        )

    @controlled_notif_button.on_click
    def _(_) -> None:
        """Show controlled notification when the button is clicked."""
        controlled_notif = server.gui.add_notification(
                            title="Controlled notification",
                            body="This cannot be closed by the user and is controlled in code only!",
                            loading=False,
                            type="controlled"
                        )
        
        @close_controlled_notif.on_click
        def _(_) -> None:
            """Clear controlled notification. """
            controlled_notif.clear()

    @loading_notif_button.on_click
    def _(_) -> None:
        """Show loading notification when the button is clicked."""
        loading_notif = server.gui.add_notification(
                        title="Loading notification",
                        body="This indicates that some action is in progress!",
                        loading=True,
                        type="persistent"
                    )
        
        time.sleep(3.0)
        loading_notif.update(title="Update notification", 
                             body="This notification was updated!")

    @clear_all_notif.on_click
    def _(_) -> None:
        """Clear all open notifications."""
        server.gui.clear_all_notification()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
