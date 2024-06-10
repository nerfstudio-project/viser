"""Gaussian splatting"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import numpy as onp
import numpy.typing as onpt
import tyro
import viser
from viser import transforms as tf


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: onpt.NDArray[onp.float32]
    """(N, 3)."""
    rgbs: onpt.NDArray[onp.float32]
    """(N, 3). Range [0, 1]."""
    opacities: onpt.NDArray[onp.float32]
    """(N, 1). Range [0, 1]."""
    covariances: onpt.NDArray[onp.float32]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian
    print(f"{num_gaussians=}")

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = onp.frombuffer(splat_buffer, dtype=onp.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(onp.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = onp.array([tf.SO3(wxyz).as_matrix() for wxyz in wxyzs])
    covariances = onp.einsum(
        "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        centers = splat_uint8[:, 0:12].copy().view(onp.float32)
        centers -= onp.mean(centers, axis=0, keepdims=True)
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def main(splat_paths: tuple[Path, ...], test_multisplat: bool = False) -> None:
    server = viser.ViserServer(share=True)
    server.gui.configure_theme(dark_mode=True)
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    for i, splat_path in enumerate(splat_paths):
        splat_data = load_splat_file(splat_path, center=True)
        server.scene.add_transform_controls(f"/{i}")
        server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
