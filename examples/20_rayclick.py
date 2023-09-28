"""Rayclicks

Visualize a mesh. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
import typing
from pathlib import Path

import trimesh
import numpy as onp

import viser
import viser.transforms as tf

server = viser.ViserServer()

mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
assert isinstance(mesh, trimesh.Trimesh)
mesh.apply_scale(0.05)

mesh_handle = server.add_mesh_trimesh(
    name="/mesh",
    mesh=mesh,
    wxyz=tf.SO3.exp(onp.array([onp.pi / 2, 0.0, 0.0])).wxyz,
    position=(0.0, 0.0, 0.0),
)

button_handle = server.add_gui_checkbox("Enable Rayclicks", False)
@button_handle.on_update
def _(_) -> None:
    # is there a better name for this?
    server.scene_pointer_enabled = button_handle.value

# Note: Scene clicks don't interrupt the scenenodeclicks.
@mesh_handle.on_click
def _(_):
    print("Mesh clicked")

hit_pos_handle = None

@server.on_scene_pointer
def on_rayclick(message: viser.ScenePointerEvent) -> None:
    global hit_pos_handle

    # check for intersection with the mesh
    mesh_tf = tf.SO3(mesh_handle.wxyz).inverse().as_matrix()
    origin = (mesh_tf @ onp.array(message.ray_origin)).reshape(1, 3)
    direction = (mesh_tf @ onp.array(message.ray_direction)).reshape(1, 3)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hit_pos, _, _ = intersector.intersects_location(origin, direction)

    # if no hit, remove the hit vis from the scene.
    if len(hit_pos) == 0:
        if hit_pos_handle is not None:
            hit_pos_handle.remove()
            hit_pos_handle = None
        return

    # get the first hit position
    hit_pos = sorted(hit_pos, key=lambda x: onp.linalg.norm(x - origin))[0]

    # put the hit position back into the world frame
    hit_pos = (tf.SO3(mesh_handle.wxyz).as_matrix() @ hit_pos.T).T
    hit_pos_mesh = trimesh.creation.icosphere(radius=0.1)
    hit_pos_mesh.vertices += hit_pos
    hit_pos_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)
    hit_pos_handle = server.add_mesh_trimesh(
        name="/hit_pos",
        mesh=hit_pos_mesh
    )

while True:
    time.sleep(10.0)