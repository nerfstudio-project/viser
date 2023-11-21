"""Mesh click events

Click on meshes to select them. The index of the last clicked mesh is displayed in the GUI.
"""

import time

import matplotlib
import numpy as onp
import trimesh.creation

import viser


def main() -> None:
    grid_shape = (4, 5)
    server = viser.ViserServer()

    with server.add_gui_folder("Last clicked"):
        x_value = server.add_gui_number(
            label="x",
            initial_value=0,
            disabled=True,
            hint="x coordinate of the last clicked mesh",
        )
        y_value = server.add_gui_number(
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
            adding_mesh = "unknown"

            if counter == 0 or counter == 1:
                adding_mesh = "box"
            else:
                adding_mesh = "icosphere"

            colors = (0.5, 0.5, 0.5)

            if counter != 0:
                index = (i * grid_shape[1] + j) / (grid_shape[0] * grid_shape[1])
                colors = colormap(index)

            handle = None

            if adding_mesh == "box":
                handle = server.add_box(
                    name=f"/sphere_{i}_{j}",
                    position=(i, j, 0.0),
                    colors=colors,
                    dimensions=(0.5, 0.5, 0.5),
                )
            elif adding_mesh == "icosphere":
                handle = server.add_icosphere(
                    name=f"/sphere_{i}_{j}",
                    position=(i, j, 0.0),
                    colors=colors,
                    radius=0.4,
                    subdivisions=2,
                )

            assert handle is not None

            @handle.on_click
            def _(_) -> None:
                x_value.value = i
                y_value.value = j

                # The new mesh will replace the old one because the names (/sphere_{i}_{j}) are
                # the same.
                create_mesh((counter + 1) % 3)

        create_mesh(0)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            add_swappable_mesh(i, j)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
