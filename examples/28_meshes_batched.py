"""Batched Meshes

Visualize batched meshes. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path
from typing import cast

import numpy as np
import trimesh

import viser
import viser.transforms as tf


def create_random_transforms(num_instances: int) -> tuple[np.ndarray, np.ndarray]:
    """Create random positions and rotations for mesh instances.

    Args:
        num_instances: Number of mesh instances to create transforms for.

    Returns:
        tuple containing:
            - positions: (N, 3) float32 array of random positions
            - rotations: (N, 4) float32 array of quaternions (wxyz format)
    """
    positions = (np.random.rand(num_instances, 3) * 2 - 1).astype(np.float32)
    rotations = np.array(
        [tf.SO3.identity().wxyz for _ in range(num_instances)],
        dtype=np.float32,
    )

    return positions, rotations


def main():
    # Load and prepare mesh data.
    mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_scale(0.002)

    mesh = trimesh.load_mesh("source/car_glb.glb")
    assert isinstance(mesh, trimesh.Scene)
    mesh.apply_scale(0.1)

    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_translation(-mesh.centroid)

    server = viser.ViserServer()

    # Add GUI controls.
    wiggle_handle = server.gui.add_checkbox("wiggle", initial_value=False)
    num_insts_handle = server.gui.add_slider(
        "num_insts", min=1, max=1000, step=1, initial_value=100
    )

    # Initialize transforms.
    positions, rotations = create_random_transforms(num_insts_handle.value)

    # Create batched mesh visualization.
    mesh_handle = server.scene.add_batched_meshes_trimesh(
        name="dragon",
        mesh=mesh,
        batched_positions=positions,
        batched_wxyzs=rotations,
        lod="auto",
    )

    # Animation loop.
    while True:
        current_num_instances = num_insts_handle.value
        update_visualization = False

        # Recreate transforms if instance count changed.
        if positions.shape[0] != current_num_instances:
            positions, rotations = create_random_transforms(current_num_instances)
            update_visualization = True

        # Add small random perturbations, to test the update latency.
        if wiggle_handle.value:
            delta = np.random.rand(current_num_instances, 3) * 0.02 - 0.01
            positions = (positions + delta).astype(np.float32)
            update_visualization = True

        # Update visualization -- positions and wxyzs together, to make sure the shapes remain consistent.
        if update_visualization:
            with server.atomic():
                mesh_handle.batched_positions = positions
                mesh_handle.batched_wxyzs = rotations

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
