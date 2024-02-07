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

import viser
import viser.transforms as tf

server = viser.ViserServer()
server.set_up_direction("+y")

mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
assert isinstance(mesh, trimesh.Trimesh)
mesh.apply_scale(0.05)

mesh_handle = server.add_mesh_trimesh(
    name="/mesh",
    mesh=mesh,
    position=(0.0, 0.0, 0.0),
)

hit_pos_handles: List[viser.GlbHandle] = []

# Button to add spheres; when clicked, we add a scene pointer event listener.
add_button_handle = server.add_gui_button("Add sphere")


@add_button_handle.on_click
def _(_):
    global mesh_handle
    add_button_handle.disabled = True

    @server.on_scene_click
    def scene_click_cb(message: viser.ScenePointerEvent) -> None:
        global mesh_handle
        # Check for intersection with the mesh, using trimesh's ray-mesh intersection.
        # Note that mesh is in the mesh frame, so we need to transform the ray.
        if message.event == "click":
            R_world_mesh = tf.SO3(mesh_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            origin = (R_mesh_world @ onp.array(message.ray_origin)).reshape(1, 3)
            direction = (R_mesh_world @ onp.array(message.ray_direction)).reshape(1, 3)
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

        elif message.event == "box":
            camera = message.client.camera

            # Put the mesh in the camera frame.
            R_world_mesh = tf.SO3(mesh_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            R_camera_world = tf.SE3.from_rotation_and_translation(tf.SO3(camera.wxyz), camera.position).inverse()
            vertices = mesh.vertices
            vertices = (R_mesh_world.as_matrix() @ vertices.T).T
            vertices = (
                R_camera_world.as_matrix() @ 
                onp.hstack([vertices, onp.ones((vertices.shape[0], 1))]).T
                ).T[:, :3]

            # Get the camera intrinsics, and project the vertices onto the image plane.
            # Initially, the screen coordinate system is:
            # (-x, -y) --> (+x, +y)
            #    |            |
            #    v            v
            # (-x, +y) --> (+x, +y)
            # ... doesn't match the NDC coordinates! This is in OpenCV's coordinate system.
            fov, aspect = camera.fov, camera.aspect
            vertices_proj = vertices[:, :2] / vertices[:, 2].reshape(-1, 1)
            vertices_proj /= onp.tan(fov/2)
            vertices_proj[:, 0] /= aspect

            # Flip the y-axis to match the NDC coordinates.
            vertices_proj[:, 1] = -vertices_proj[:, 1]

            # Select the vertices that lie inside the 2D selected box, once projected.
            mask = (
                (vertices_proj > onp.array(message.screen_pos[0])) & 
                (vertices_proj < onp.array(message.screen_pos[1]))
                ).all(axis=1)[..., None]

            # Update the mesh color based on whether the vertices are inside the box
            mesh.visual.vertex_colors = onp.where(
                mask, (1.0, 0.0, 0.0, 1.0), (0.9, 0.9, 0.9, 1.0)
            )
            mesh_handle = server.add_mesh_trimesh(
                name="/mesh",
                mesh=mesh,
                position=(0.0, 0.0, 0.0),
            )

            add_button_handle.disabled = False
            server.remove_scene_click_callback(scene_click_cb)


# Button to clear spheres
clear_button_handle = server.add_gui_button("Clear spheres")


@clear_button_handle.on_click
def _(_):
    """Reset the mesh color and remove all click-generated spheres."""
    global mesh_handle
    for handle in hit_pos_handles:
        handle.remove()
    hit_pos_handles.clear()
    mesh.visual.vertex_colors = (0.9, 0.9, 0.9, 1.0)
    mesh_handle = server.add_mesh_trimesh(
        name="/mesh",
        mesh=mesh,
        position=(0.0, 0.0, 0.0),
    )


while True:
    time.sleep(10.0)
