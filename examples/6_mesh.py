import time
from pathlib import Path

import numpy as np
import trimesh

import viser

mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
assert isinstance(mesh, trimesh.Trimesh)

vertices = mesh.vertices
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()

while True:
    server.add_mesh(name="/dragon", vertices=vertices, faces=faces)
    time.sleep(10.0)
