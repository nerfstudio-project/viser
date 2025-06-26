"""Batched mesh rendering

Efficiently render many instances of the same mesh with different transforms.

This example demonstrates batched mesh rendering, which is essential for visualizing large numbers of similar objects like particles, forest scenes, or crowd simulations. Batched rendering is dramatically more efficient than creating individual scene objects.

**Key features:**

* :meth:`viser.SceneApi.add_batched_meshes_simple` for instanced mesh rendering
* :meth:`viser.SceneApi.add_batched_axes` for coordinate frame instances
* Per-instance transforms (position, rotation, scale)
* Level-of-detail (LOD) optimization for performance
* Real-time animation of instance properties

Batched meshes have some limitations: GLB animations are not supported, hierarchy is flattened, and each mesh in a GLB is instanced separately. However, they excel at rendering thousands of objects efficiently.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh

import viser


def create_grid_transforms(
    num_instances: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grid positions, rotations, and scales for mesh instances."""
    grid_size = int(np.ceil(np.sqrt(num_instances)))

    # Create grid positions
    x = np.arange(grid_size) - (grid_size - 1) / 2
    y = np.arange(grid_size) - (grid_size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    positions = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    positions[:, 0] = xx.flatten()
    positions[:, 1] = yy.flatten()
    positions[:, 2] = 1.0
    positions = positions[:num_instances]

    # All instances have identity rotation
    rotations = np.zeros((num_instances, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1

    # Initial scales.
    scales = np.linalg.norm(positions, axis=-1)
    scales = np.sin(scales * 1.5) * 0.5 + 1.0
    return positions, rotations, scales.astype(np.float32)


def main():
    # Load and prepare mesh data.
    dragon_mesh = trimesh.load_mesh(str(Path(__file__).parent / "../assets/dragon.obj"))
    assert isinstance(dragon_mesh, trimesh.Trimesh)
    dragon_mesh.apply_scale(0.005)
    dragon_mesh.vertices -= dragon_mesh.centroid

    dragon_mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
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

    animate_checkbox = server.gui.add_checkbox("Animate", initial_value=True)
    per_axis_scale_checkbox = server.gui.add_checkbox(
        "Per-axis scale during animation", initial_value=True
    )
    lod_checkbox = server.gui.add_checkbox("Enable LOD", initial_value=True)
    cast_shadow_checkbox = server.gui.add_checkbox("Cast shadow", initial_value=True)

    # Initialize transforms.
    positions, rotations, scales = create_grid_transforms(instance_count_slider.value)

    # Create batched mesh visualization.
    axes_handle = server.scene.add_batched_axes(
        name="axes",
        batched_positions=positions,
        batched_wxyzs=rotations,
        batched_scales=scales,
    )
    mesh_handle = server.scene.add_batched_meshes_simple(
        name="dragon",
        vertices=dragon_mesh.vertices,
        faces=dragon_mesh.faces,
        batched_positions=positions,
        batched_wxyzs=rotations,
        batched_scales=scales,
        lod="auto",
    )

    # Animation loop.
    while True:
        n = instance_count_slider.value

        # Update props based on GUI controls.
        mesh_handle.lod = "auto" if lod_checkbox.value else "off"
        mesh_handle.cast_shadow = cast_shadow_checkbox.value

        # Recreate transforms if instance count changed.
        if positions.shape[0] != n:
            positions, rotations, scales = create_grid_transforms(n)
            grid_size = int(np.ceil(np.sqrt(n)))

            with server.atomic():
                # Update grid size.
                grid_handle.width = grid_handle.height = grid_size + 2
                grid_handle.width_segments = grid_handle.height_segments = grid_size + 2

                # Update all transforms.
                mesh_handle.batched_positions = axes_handle.batched_positions = (
                    positions
                )
                mesh_handle.batched_wxyzs = axes_handle.batched_wxyzs = rotations
                mesh_handle.batched_scales = axes_handle.batched_scales = scales

        # Animate if enabled.
        elif animate_checkbox.value:
            # Animate positions.
            positions[:, :2] += np.random.uniform(-0.01, 0.01, (n, 2))

            # Animate scales with wave effect.
            if per_axis_scale_checkbox.value:
                t = time.perf_counter() * 2.0
                scales = np.linalg.norm(positions, axis=-1)
                scales = np.stack(
                    [
                        np.sin(scales * 1.5 - t) * 0.5 + 1.0,
                        np.sin(scales * 1.5 - t + np.pi / 2.0) * 0.5 + 1.0,
                        np.sin(scales * 1.5 - t + np.pi) * 0.5 + 1.0,
                    ],
                    axis=-1,
                )
                assert scales.shape == (n, 3)
            else:
                t = time.perf_counter() * 2.0
                scales = np.linalg.norm(positions, axis=-1)
                scales = np.sin(scales * 1.5 - t) * 0.5 + 1.0
                assert scales.shape == (n,)

            with server.atomic():
                mesh_handle.batched_positions = positions
                mesh_handle.batched_scales = scales
                axes_handle.batched_positions = positions
                axes_handle.batched_scales = scales

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
