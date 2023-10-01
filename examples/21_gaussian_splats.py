"""Gaussian splatting
"""

import time
from pathlib import Path

import numpy as onp
import tyro

import viser
from viser import transforms as tf


def main(splat_path: Path) -> None:
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
    scales = splat_uint8[:, 12:24].view(onp.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = onp.array([tf.SO3(wxyz).as_matrix() for wxyz in wxyzs])
    covariances = onp.einsum(
        "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )

    server = viser.ViserServer()
    server.configure_theme(dark_mode=True)
    gui_reset_up = server.add_gui_button(
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

    server.add_gaussian_splats(
        "/gaussian_splats",
        # Centers should have shape (N, 3).
        centers=splat_uint8[:, 0:12].view(onp.float32),
        # Colors should have shape (N, 4).
        rgbs=splat_uint8[:, 24:27] / 255.0,
        opacities=splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        covariances=covariances,
    )

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
