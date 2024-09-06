"""Set up direction

`.set_up_direction()` can help us set the global up direction."""

import time

import viser


def main() -> None:
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    gui_up = server.gui.add_vector3(
        "Up Direction",
        initial_value=(0.0, 0.0, 1.0),
        step=0.01,
    )

    @gui_up.on_update
    def _(_) -> None:
        server.scene.set_up_direction(gui_up.value)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
