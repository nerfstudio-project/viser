"""Meshes

Visualize a mesh. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as onp
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
# server.scene.enable_default_lights(enabled=False)
server.scene.set_environment_map(name="something", hdri="hdri/goegap_split.hdr")
server.scene.add_mesh_simple(
    name="/simple",
    vertices=vertices,
    faces=faces,
    wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
    position=(0.0, 0.0, 0.0),
)
server.scene.add_mesh_trimesh(
    name="/trimesh",
    mesh=mesh.smoothed(),
    wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
    position=(0.0, 5.0, 0.0),
)
# server.scene.add_light_directional(
#     name="directional light", color=0xFF0000, position=(1.0, 1.0, 0.0), intensity=2.0
# )
# server.scene.add_light_ambient(name="ambient light", color=0x003333)
# server.scene.add_light_directional(
#     name="poopy light", color=0x0000FF, position=(0.0, 0.0, 1.0), intensity=0.3
# )
# server.scene.add_light_hemisphere(
#     name="hemisphere light", skyColor=0x00ff00, groundColor=0x0000ff)
# server.scene.add_light_point(name="point light", color=0x440000, position = (0,0,1), distance = 30, power = 30)
# server.scene.add_light_rectarea(name="rectangular area light", position = (0,0,3), power = 50, intensity = 20)
# server.scene.add_light_spot(
#     name="spot light", color=0xDDDD55, position=(0, 1, 3), intensity=10, distance=20
# )


while True:
    time.sleep(10.0)
