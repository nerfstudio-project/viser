"""Click callback demonstration

Click on meshes to select them. The index of the last clicked mesh is displayed in the GUI.
"""

import time
from pathlib import Path

import numpy as onp
import trimesh as tr
import matplotlib

import viser
import viser.transforms as tf

grid_shape = (4, 5)
server = viser.ViserServer()

colormap = matplotlib.colormaps['tab20']

def swap_mesh(i, j):
    """
    Simple callback that swaps between:
     - a gray box
     - a colored box 
     - a colored sphere 
    
    Color is chosen based on the position (i, j) of the mesh in the grid
    """
    def create_handle():
        nonlocal curr_state, handle

        if curr_state == 0:
            mesh = tr.creation.box((0.5, 0.5, 0.5))
        elif curr_state == 1:
            mesh = tr.creation.box((0.5, 0.5, 0.5))
        else:
            mesh = tr.creation.icosphere(subdivisions=2, radius=0.4)

        colors = colormap(
            (i*grid_shape[1] + j + onp.random.rand(mesh.vertices.shape[0])) / (grid_shape[0]*grid_shape[1])
        )
        if curr_state != 0:
            mesh.visual.vertex_colors = colors

        handle = server.add_mesh_trimesh(
            name=f"/sphere_{i}_{j}",
            mesh=mesh,
            clickable=True,
            position=(i, j, 0.0),
        )
        handle.on_click(on_click)
        curr_state = (curr_state + 1) % 3
    
    def on_click(_):
        nonlocal handle
        x_value.value = i
        y_value.value = j
        handle.remove()
        create_handle()

    curr_state, handle = 0, None
    create_handle()

with server.gui_folder("Last clicked"):
    x_value = server.add_gui_number(
        name="x", 
        initial_value=0, 
        disabled=True,
        hint="x coordinate of the last clicked mesh"
    )
    y_value = server.add_gui_number(
        name="y",
        initial_value=0,
        disabled=True,
        hint="y coordinate of the last clicked mesh"
    )

for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        swap_mesh(i, j)

while True:
    time.sleep(10.0)