"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import numpy as onp
import numpy.typing as onpt
import tyro
import viser
from plyfile import PlyData
from viser import transforms as tf


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: onpt.NDArray[onp.floating]
    """(N, 3)."""
    rgbs: onpt.NDArray[onp.floating]
    """(N, 3). Range [0, 1]."""
    opacities: onpt.NDArray[onp.floating]
    """(N, 1). Range [0, 1]."""
    covariances: onpt.NDArray[onp.floating]
    """(N, 3, 3)."""


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
    splat_uint8 = onp.frombuffer(splat_buffer, dtype=onp.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(onp.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = onp.array([tf.SO3(wxyz).as_matrix() for wxyz in wxyzs])
    covariances = onp.einsum(
        "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(onp.float32)
    if center:
        centers -= onp.mean(centers, axis=0, keepdims=True)
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
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = onp.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = onp.exp(onp.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = onp.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * onp.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1) 
    # print(v["f_dc_0"].shape) # prints (numGaussians)
    # print(colors.shape) # prints (numGaussians, 3)
    opacities = 1.0 / (1.0 + onp.exp(-v["opacity"][:, None]))
    
    # Load all zero order SH coefficients
    dc_terms = onp.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)

    # Load higher order SH coefficients (f_rest_0, ... f_rest_44), which are either level 1 or higher
    # Note: .ply file supports maximum SH degree of 3, R = 0 (mod 3), G = 1 (mod 3), B = 2 (mod 3)
    rest_terms = []
    i = 0
    while f"f_rest_{i}" in v:
        rest_terms.append(v[f"f_rest_{i}"])
        i += 1
    if len(rest_terms) > 0: # if we do have higher than zero order SH, we will process them and add them here.
        sh_coeffs = onp.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]] + rest_terms, axis=1)
    sh_degree = int(onp.sqrt(sh_coeffs.shape[1] // 3) - 1)

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = onp.einsum(
        "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= onp.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)

    print(sh_coeffs.shape)
    print(v["x"].shape)

    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
        "sh_degree": sh_degree,
        "sh_coeffs": sh_coeffs,
    }


def main(splat_paths: tuple[Path, ...]) -> None:
    server = viser.ViserServer()
    print(server.request_share_url())
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
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=True)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene._add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
            sh_degree=splat_data["sh_degree"],
            sh_coefficients=splat_data["sh_coeffs"],
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



print("yapyap")