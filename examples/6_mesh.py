"""Visualize mesh. To get the demo data, checkout assets/download_dragon_mesh.sh."""

import time
from pathlib import Path
from typing import Tuple

import numpy as onp
import trimesh
from scipy.spatial.transform import Rotation

import viser


def quat_from_so3(
    omega: Tuple[float, float, float]
) -> Tuple[float, float, float, float]:
    xyzw = Rotation.from_rotvec(onp.array(omega)).as_quat()
    return (xyzw[3], xyzw[0], xyzw[1], xyzw[2])


mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
assert isinstance(mesh, trimesh.Trimesh)

vertices = mesh.vertices * 0.5
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()
server.add_frame(
    name="/frame",
    wxyz=quat_from_so3((onp.pi / 2, 0.0, 0.0)),
    position=(0.0, 0.0, 0.0),
    show_axes=False,
)
server.add_mesh(name="/frame/dragon", vertices=vertices, faces=faces)

while True:
    time.sleep(10.0)
