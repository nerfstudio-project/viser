"""Batched Meshes

Visualize batched meshes. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as np
import trimesh

import viser
import viser.transforms as tf


def create_grid_transforms(num_instances: int) -> tuple[np.ndarray, np.ndarray]:
    """Create grid positions and rotations for mesh instances.

    Args:
        num_instances: Number of mesh instances to create transforms for.

    Returns:
        tuple containing:
            - positions: (N, 3) float32 array of random positions
            - rotations: (N, 4) float32 array of quaternions (wxyz format)
    """
    grid_size = int(np.ceil(np.sqrt(num_instances)))
    x = (np.arange(grid_size) - 0.5 * (grid_size - 1))
    y = (np.arange(grid_size) - 0.5 * (grid_size - 1))
    positions = np.stack(np.meshgrid(x, y, 1.0), axis=-1).reshape(-1, 3)
    positions = positions[:num_instances]
    rotations = np.array(
        [tf.SO3.identity().wxyz for _ in range(num_instances)],
        dtype=np.float32,
    )

    return positions, rotations


def main():
    # Load and prepare mesh data.
    dragon_mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
    assert isinstance(dragon_mesh, trimesh.Trimesh)
    dragon_mesh.apply_scale(0.005)
    dragon_mesh.vertices -= dragon_mesh.centroid

    dragon_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    dragon_mesh.apply_translation(-dragon_mesh.centroid)

    server = viser.ViserServer()
    server.scene.configure_default_lights()
    grid_handle = server.scene.add_grid(
        name="grid",
        width=12,
        height=12,
        width_segments=12,
        height_segments=12,
    )

    # Add GUI controls.
    instance_count_slider = server.gui.add_slider(
        "# of instances", min=1, max=1000, step=1, initial_value=100
    )

    # Wiggle mesh, to test pose update latency.
    wiggle_checkbox = server.gui.add_checkbox("Wiggle", initial_value=False)

    # Allow user to toggle LOD.
    lod_checkbox = server.gui.add_checkbox("Enable LoD", initial_value=True)
    @lod_checkbox.on_update
    def _(_):
        mesh_handle.lod = "auto" if lod_checkbox.value else "off"

    # Allow user to toggle cast shadow.
    cast_shadow_checkbox = server.gui.add_checkbox("Cast shadow", initial_value=True)
    @cast_shadow_checkbox.on_update
    def _(_):
        mesh_handle.cast_shadow = cast_shadow_checkbox.value

    # Initialize transforms.
    positions, rotations = create_grid_transforms(instance_count_slider.value)

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
            positions, rotations = create_grid_transforms(current_instance_count)
            update_visualization = True

        # Add small random perturbations, to test the update latency.
        if wiggle_checkbox.value:
            delta = np.random.rand(current_instance_count, 2) * 0.02 - 0.01
            positions[:, :2] = (positions[:, :2] + delta).astype(np.float32)
            update_visualization = True

        # Update visualization -- positions and wxyzs together, to make sure the shapes remain consistent.
        if update_visualization:
            with server.atomic():
                mesh_handle.batched_positions = positions
                mesh_handle.batched_wxyzs = rotations

                grid_size = int(np.ceil(np.sqrt(current_instance_count)))
                grid_handle.width = grid_size + 2
                grid_handle.height = grid_size + 2
                grid_handle.width_segments = grid_size + 2
                grid_handle.height_segments = grid_size + 2

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
