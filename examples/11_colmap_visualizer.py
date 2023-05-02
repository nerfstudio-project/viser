"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs.
"""

import time
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as onp
import tyro
from tqdm.auto import tqdm

import viser
import viser.transforms as tf
from viser.extras.colmap import read_cameras_binary, read_images_binary, read_points3d_binary


def main(
    colmap_path: Path = Path(__file__).parent / "assets/colmap_sparse",
    images_path: Path = Path(__file__).parent / "assets/colmap_images",
    downsample_factor: int = 2,
    max_points: Optional[int] = 100000,
    max_frames: Optional[int] = 100,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        downsample_factor: Downsample factor for the images.
        max_points: Maximum number of points to visualize.
        max_frames: Maximum number of frames to visualize.
    """
    server = viser.ViserServer()

    # Load the colmap info.
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")

    # Set a world rotation to make the scene upright.
    server.add_frame(
        "/colmap",
        wxyz=tf.SO3.exp(onp.array([-onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    # Set the point cloud.
    position = onp.array([points3d[p_id].xyz for p_id in points3d])
    color = onp.array([points3d[p_id].rgb for p_id in points3d])
    if max_points is not None:
        onp.random.shuffle(position)
        onp.random.shuffle(color)
        position = position[:max_points]
        color = color[:max_points]
    server.add_point_cloud(name="/colmap/pcd", position=position, color=color, point_size=0.01)

    # Interpret the images and cameras.
    img_ids = [im.id for im in images.values()]
    if max_frames is not None:
        onp.random.shuffle(img_ids)
        img_ids = sorted(img_ids[:max_frames])

    for img_id in tqdm(img_ids):
        cam = cameras[img_id]
        assert cam.id == img_id
        img = images[img_id]

        image_filename = images_path / img.name
        # if doesn't exist, then skip
        if not image_filename.exists():
            continue

        rot = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)
        bottom = onp.array([0, 0, 0, 1.]).reshape(1, 4)
        w2c = onp.concatenate([onp.concatenate([rot, t], 1), bottom], 0)
        c2w = onp.linalg.inv(w2c)
        server.add_frame(f"/colmap/frames/t{img_id}", show_axes=False)
        server.add_frame(
            f"/colmap/frames/t{img_id}/camera",
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            axes_length=0.1,
            axes_radius=0.005,
        )
        # TODO: set these properly
        H, W = 480, 640
        focallen = 500
        aspect = W / H
        scale = 0.15
        server.add_camera_frustum(
            f"/colmap/frames/t{img_id}/camera/frustum",
            fov=2 * onp.arctan2(H / 2, focallen),
            aspect=aspect,
            scale=scale,
        )
        image = iio.imread(image_filename)
        image = image[::downsample_factor, ::downsample_factor]
        # flip verticle dimension
        image = image[::-1, :]
        server.add_image(
            f"/colmap/frames/t{img_id}/camera/image",
            image,
            render_width=scale * aspect,
            render_height=scale,
            format="png",
        )

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
