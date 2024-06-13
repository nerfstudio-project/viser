"""Notifications

Examples of adding notifications in Viser."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()

    gui_reset_scene = server.gui.add_button("Reset Scene")
    server.gui.add_notification(
        title="test",
        body="testing",
        autoClose=True,
        withCloseButton=True,
        loading=False,
    )

    @gui_reset_scene.on_click
    def _(_) -> None:
        """Reset the scene when the reset button is clicked."""
        # server.gui.add_notification()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
