import time
from pathlib import Path

import trimesh

import viser

mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
assert isinstance(mesh, trimesh.Trimesh)

vertices = mesh.vertices
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()
server.add_mesh(name="/dragon", vertices=vertices, faces=faces)

while True:
    time.sleep(10.0)
