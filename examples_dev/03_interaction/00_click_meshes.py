"""Mesh click events

Click on meshes to select them. The index of the last clicked mesh is displayed in the GUI.
"""

import time

import matplotlib

import viser


def main() -> None:
    grid_shape = (4, 5)
    server = viser.ViserServer()

    with server.gui.add_folder("Last clicked"):
        x_value = server.gui.add_number(
            label="x",
            initial_value=0,
            disabled=True,
            hint="x coordinate of the last clicked mesh",
        )
        y_value = server.gui.add_number(
            label="y",
            initial_value=0,
            disabled=True,
            hint="y coordinate of the last clicked mesh",
        )

    def add_swappable_mesh(i: int, j: int) -> None:
        """Simple callback that swaps between:
         - a gray box
         - a colored box
         - a colored sphere

        Color is chosen based on the position (i, j) of the mesh in the grid.
        """

        colormap = matplotlib.colormaps["tab20"]

        def create_mesh(counter: int) -> None:
            if counter == 0:
                color = (0.8, 0.8, 0.8)
            else:
                index = (i * grid_shape[1] + j) / (grid_shape[0] * grid_shape[1])
                color = colormap(index)[:3]

            if counter in (0, 1):
                handle = server.scene.add_box(
                    name=f"/sphere_{i}_{j}",
                    position=(i, j, 0.0),
                    color=color,
                    dimensions=(0.5, 0.5, 0.5),
                )
            else:
                handle = server.scene.add_icosphere(
                    name=f"/sphere_{i}_{j}",
                    radius=0.4,
                    color=color,
                    position=(i, j, 0.0),
                )

            @handle.on_click
            def _(_) -> None:
                x_value.value = i
                y_value.value = j

                # The new mesh will replace the old one because the names
                # /sphere_{i}_{j} are the same.
                create_mesh((counter + 1) % 3)

        create_mesh(0)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            add_swappable_mesh(i, j)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
