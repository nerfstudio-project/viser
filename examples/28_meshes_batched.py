"""Batched Meshes

Visualize batched meshes. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d

import viser
import viser.transforms as tf


def _decimate_mesh(mesh: trimesh.Trimesh, target_faces: int = 500) -> trimesh.Trimesh:
    """Decimate a mesh using Open3D's quartile decimation.

    Args:
        mesh: trimesh.Trimesh object containing the mesh to decimate
        target_faces: Target number of faces to keep after decimation

    Returns:
        trimesh.Trimesh: Decimated mesh
    """
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Decimate mesh
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )

    # Convert back to trimesh
    vertices_decimated = np.asarray(o3d_mesh.vertices)
    faces_decimated = np.asarray(o3d_mesh.triangles)

    return trimesh.Trimesh(vertices=vertices_decimated, faces=faces_decimated)


def main():
    mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_scale(0.01)
    # mesh = trimesh.creation.cylinder(radius=0.5, height=1.0)

    vertices = mesh.vertices
    faces = mesh.faces

    mesh_downsample_0 = _decimate_mesh(mesh, target_faces=len(faces) // 10)
    mesh_downsample_1 = _decimate_mesh(mesh, target_faces=len(faces) // 100)

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
    handle = server.scene.add_batched_meshes(
        name="dragon",
        vertices=vertices,
        faces=faces,
        batched_wxyzs=rotations,
        batched_positions=positions,
        lod_list=(
            (mesh_downsample_0.vertices, mesh_downsample_0.faces, 1),
            (mesh_downsample_1.vertices, mesh_downsample_1.faces, 4),
        ),
    )

    while True:
        # num_instances = 100
        # delta_positions = np.random.rand(num_instances, 3) * 0.01 - 0.005
        # positions = (positions + delta_positions).astype(np.float32)
        # handle.batched_positions = positions

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
