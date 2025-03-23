"""Batched Meshes

Visualize batched meshes. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

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
    dragon_mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
    assert isinstance(dragon_mesh, trimesh.Trimesh)
    dragon_mesh.apply_scale(0.002)

    dragon_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    dragon_mesh.apply_translation(-dragon_mesh.centroid)

    server = viser.ViserServer()

    # Add GUI controls.
    instance_count_slider = server.gui.add_slider(
        "# of instances", min=1, max=1000, step=1, initial_value=100
    )

    # Wiggle mesh, to test pose update latency.
    wiggle_checkbox = server.gui.add_checkbox("Wiggle", initial_value=False)

    # Allow user to toggle LOD.
    lod_checkbox = server.gui.add_checkbox("Enable LoD", initial_value=True)
    @lod_checkbox.on_update
    def update_lod(_):
        mesh_handle.lod = "auto" if lod_checkbox.value else "off"

    # Initialize transforms.
    positions, rotations = create_random_transforms(instance_count_slider.value)

    # Create batched mesh visualization.
    mesh_handle = server.scene.add_batched_meshes_trimesh(
        name="dragon",
        mesh=dragon_mesh,
        batched_positions=positions,
        batched_wxyzs=rotations,
        lod="auto" if lod_checkbox.value else "off",
    )

    # Animation loop.
    while True:
        current_instance_count = instance_count_slider.value
        update_visualization = False

        # Recreate transforms if instance count changed.
        if positions.shape[0] != current_instance_count:
            positions, rotations = create_random_transforms(current_instance_count)
            update_visualization = True

        # Add small random perturbations, to test the update latency.
        if wiggle_checkbox.value:
            delta = np.random.rand(current_instance_count, 3) * 0.02 - 0.01
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
