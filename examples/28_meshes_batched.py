"""Batched Meshes

Visualize batched meshes. To get the demo data, see `./assets/download_dragon_mesh.sh`.
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

# Create multiple instances of the mesh with different positions
num_instances = 100
positions = (
    np.random.rand(num_instances, 3) * 10 - 5
)  # Random positions in a 10x10x10 cube
rotations = [tf.SO3.from_x_radians(np.pi / 2).wxyz for _ in range(num_instances)]
positions = positions.astype(np.float32)
rotations = np.array(rotations, dtype=np.float32)

server = viser.ViserServer()
server.scene.add_batched_meshes(
    name="dragon",
    vertices=vertices,
    faces=faces,
    batched_wxyzs=rotations,
    batched_positions=positions,
)

while True:
    time.sleep(10.0)
