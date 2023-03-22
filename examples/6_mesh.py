import time
import viser

import numpy as np
import numpy.typing as npt
import trimesh


with open("examples/assets/dragon.obj", "rb") as f:
    mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(f))

vertices = np.asarray(mesh.vertices).astype(np.float32)
faces = np.asarray(mesh.faces).astype(np.uint32)
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()
server.reset_scene()

while True:
    server.add_mesh(name="/dragon", vertices_f32=vertices, faces_uint32=faces)
    time.sleep(10.0)
