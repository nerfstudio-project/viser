"""Notifications

Examples of adding notifications in Viser."""

import time

import viser

def main() -> None:
    server = viser.ViserServer()

    gui_upload_button = server.gui.add_upload_button(
                "Upload", icon=viser.Icon.UPLOAD
            )

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
