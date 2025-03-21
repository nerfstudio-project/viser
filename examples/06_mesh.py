"""Meshes

Visualize a mesh. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as np
import trimesh

import viser
import viser.transforms as tf

mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
assert isinstance(mesh, trimesh.Trimesh)
mesh.apply_scale(0.05)

vertices = mesh.vertices
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()
server.scene.add_mesh_simple(
    name="/simple",
    vertices=vertices,
    faces=faces,
    wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
    position=(0.0, 0.0, 0.0),
)
server.scene.add_mesh_trimesh(
    name="/trimesh",
    mesh=mesh,
    wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
    position=(0.0, 5.0, 0.0),
)
grid = server.scene.add_grid(
    "grid",
    width=20.0,
    height=20.0,
    position=np.array([0.0, 0.0, -2.0]),
)

while True:
    time.sleep(10.0)
