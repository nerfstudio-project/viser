"""Scene pointer events.

This example shows how to use scene pointer events to specify rays, and how they can be
used to interact with the scene (e.g., ray-mesh intersections).

To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path
from typing import List

import numpy as onp
import trimesh.creation
import trimesh.ray
import shapely

import viser
import viser.transforms as tf

server = viser.ViserServer()

mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
assert isinstance(mesh, trimesh.Trimesh)
mesh.apply_scale(0.05)

mesh_handle = server.add_mesh_trimesh(
    name="/mesh",
    mesh=mesh,
    wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
    position=(0.0, 0.0, 0.0),
)

hit_pos_handles: List[viser.GlbHandle] = []

# Button to add spheres; when clicked, we add a scene pointer event listener.
add_button_handle = server.add_gui_button("Add sphere")


@add_button_handle.on_click
def _(_):
    add_button_handle.disabled = True

    @server.on_scene_click
    def scene_click_cb(message: viser.ScenePointerEvent) -> None:
        # Check for intersection with the mesh, using trimesh's ray-mesh intersection.
        # Note that mesh is in the mesh frame, so we need to transform the ray.
        if message.event == "click":
            R_world_mesh = tf.SO3(mesh_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            origin = (R_mesh_world @ onp.array(message.ray_origin[0])).reshape(1, 3)
            direction = (R_mesh_world @ onp.array(message.ray_direction[0])).reshape(1, 3)
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            hit_pos, _, _ = intersector.intersects_location(origin, direction)

            if len(hit_pos) == 0:
                return

            # Successful click => remove callback.
            add_button_handle.disabled = False
            server.remove_scene_click_callback(scene_click_cb)

            # Get the first hit position (based on distance from the ray origin).
            hit_pos = min(hit_pos, key=lambda x: onp.linalg.norm(x - origin))

            # Create a sphere at the hit location.
            hit_pos_mesh = trimesh.creation.icosphere(radius=0.1)
            hit_pos_mesh.vertices += R_world_mesh @ hit_pos
            hit_pos_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
            hit_pos_handle = server.add_mesh_trimesh(
                name=f"/hit_pos_{len(hit_pos_handles)}", mesh=hit_pos_mesh
            )
            hit_pos_handles.append(hit_pos_handle)

        elif message.event == "scribble":
            # This function takes a while. The actual message sending isn't too bad. 
            R_world_mesh = tf.SO3(mesh_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            num_waypoints = len(message.ray_origin)
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            
            hit_pos_list = []
            for i in range(num_waypoints):
                origin = (R_mesh_world @ onp.array(message.ray_origin[i])).reshape(1, 3)
                direction = (R_mesh_world @ onp.array(message.ray_direction[i])).reshape(1, 3)
                hit_pos, _, _ = intersector.intersects_location(origin, direction)
                if len(hit_pos) == 0:
                    continue
                hit_pos = min(hit_pos, key=lambda x: onp.linalg.norm(x - origin))
                hit_pos = R_world_mesh @ hit_pos
                hit_pos_list.append(hit_pos)
            
            if len(hit_pos_list) == 0:
                return

            add_button_handle.disabled = False
            server.remove_scene_click_callback(scene_click_cb)

            square = shapely.geometry.Point(0.0, 0.0).buffer(0.02)
            try:
                hit_pos_mesh = trimesh.creation.sweep_polygon(
                    polygon=square,
                    path=onp.array(hit_pos_list),
                )
            except:
                return
            hit_pos_mesh.fix_normals()
            hit_pos_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)
            hit_pos_handle = server.add_mesh_trimesh(
                name=f"/hit_pos_{len(hit_pos_handles)}", mesh=hit_pos_mesh
            )
            hit_pos_handles.append(hit_pos_handle)


# Button to clear spheres
clear_button_handle = server.add_gui_button("Clear spheres")


@clear_button_handle.on_click
def _(_):
    for handle in hit_pos_handles:
        handle.remove()
    hit_pos_handles.clear()


while True:
    time.sleep(10.0)
