"""Set Up Direction

`.set_up_direction()` can help us set the global up direction."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()
    server.world_axes.visible = True
    gui_up = server.add_gui_vector3(
        "Up Direction",
        value=(0.0, 0.0, 1.0),
        step=0.01,
    )

    @gui_up.on_update
    def _(_) -> None:
        server.set_up_direction(gui_up.value)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
