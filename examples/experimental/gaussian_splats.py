"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData

import viser
from viser import transforms as tf


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""
    sh_coeffs: npt.NDArray[np.floating]


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
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

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
        # No SH coefficients in the splat file.
        "sh_coeffs": np.zeros((num_gaussians, 45), dtype=np.float32),
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))
    dc_coeffs = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    # Rest coefficients 0-14 belongs to RED channel, 15-29 to GREEN, 30-44 to BLUE
    # Due to spherical harmonic calculations calculating a triplet at a time
    # we need to stack them by (0,15,30), (1,16,31), ..., (14,29,44)
    rest_coeffs = []
    for i in range(15):
        rest_coeffs.append(v[f"f_rest_{i}"])
        rest_coeffs.append(v[f"f_rest_{i + 15}"])
        rest_coeffs.append(v[f"f_rest_{i + 30}"])
    rest_coeffs = np.stack(rest_coeffs, axis=1)
    sh_coeffs = np.concatenate([dc_coeffs, rest_coeffs], axis=1)

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
        "sh_coeffs": sh_coeffs,
    }


def main(splat_paths: tuple[Path, ...]) -> None:
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=True)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
            sh_coeffs=splat_data["sh_coeffs"],
        )

        remove_button = server.gui.add_button(f"Remove splat object {i}")

        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
