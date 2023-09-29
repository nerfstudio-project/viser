"""Scene pointer events.

This example shows how to use scene pointer events to specify rays, 
and how they can be used to interact with the scene (e.g., ray-mesh intersections).

To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from typing import List
from pathlib import Path

import trimesh.ray
import trimesh.creation
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

hit_pos_handles: List[viser.GlbHandle] = []

# Button to add spheres; when clicked, we add a scene pointer event listener
add_button_handle = server.add_gui_button("Add sphere")


@add_button_handle.on_click
def _(_):
    add_button_handle.disabled = True

    @server.on_scene_pointer_event
    def on_rayclick(message: viser.ScenePointerEvent) -> None:
        # Check for intersection with the mesh, using trimesh's ray-mesh intersection
        # Note that mesh is in the mesh frame, so we need to transform the ray
        mesh_tf = tf.SO3(mesh_handle.wxyz).inverse().as_matrix()
        origin = (mesh_tf @ onp.array(message.ray_origin)).reshape(1, 3)
        direction = (mesh_tf @ onp.array(message.ray_direction)).reshape(1, 3)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        hit_pos, _, _ = intersector.intersects_location(origin, direction)

        if len(hit_pos) == 0:
            return

        # get the first hit position (based on distance from the ray origin)
        hit_pos = sorted(hit_pos, key=lambda x: onp.linalg.norm(x - origin))[0]

        # Put the hit position back into the world frame
        hit_pos = (tf.SO3(mesh_handle.wxyz).as_matrix() @ hit_pos.T).T
        hit_pos_mesh = trimesh.creation.icosphere(radius=0.1)
        hit_pos_mesh.vertices += hit_pos
        hit_pos_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
        hit_pos_handle = server.add_mesh_trimesh(
            name=f"/hit_pos_{len(hit_pos_handles)}", mesh=hit_pos_mesh
        )
        hit_pos_handles.append(hit_pos_handle)
        server.remove_scene_pointer_event(on_rayclick)
        add_button_handle.disabled = False


# Button to clear spheres
clear_button_handle = server.add_gui_button("Clear spheres")


@clear_button_handle.on_click
def _(_):
    global hit_pos_handles
    for handle in hit_pos_handles:
        handle.remove()
    hit_pos_handles = []


while True:
    time.sleep(10.0)
