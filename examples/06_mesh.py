"""Meshes

Visualize a mesh. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as onp
import trimesh

import viser
import viser.transforms as tf

mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
assert isinstance(mesh, trimesh.Trimesh)

vertices = mesh.vertices * 0.5
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()
server.add_mesh(
    name="/frame",
    vertices=vertices,
    faces=faces,
    wxyz=tf.SO3.exp(onp.array([onp.pi / 2, 0.0, 0.0])).wxyz,
    position=(0.0, 0.0, 0.0),
)

while True:
    time.sleep(10.0)
