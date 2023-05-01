"""RealSense visualizer

Connect to a RealSense camera, then visualize RGB-D readings as a point clouds. Requires
pyrealsense2.
"""
import contextlib
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs  # type: ignore
from tqdm.auto import tqdm

import viser


@contextlib.contextmanager
def realsense_pipeline(fps: int = 30):
    """Context manager that yields a RealSense pipeline."""

    # Configure depth and color streams.
    pipeline = rs.pipeline()  # type: ignore
    config = rs.config()  # type: ignore

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)  # type: ignore
    config.resolve(pipeline_wrapper)

    config.enable_stream(rs.stream.depth, rs.format.z16, fps)  # type: ignore
    config.enable_stream(rs.stream.color, rs.format.rgb8, fps)  # type: ignore

    # Start streaming.
    pipeline.start(config)

    yield pipeline

    # Close pipeline when done.
    pipeline.close()


def point_cloud_arrays_from_frames(
    depth_frame, color_frame
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Maps realsense frames to two arrays.

    Returns:
    - A point position array: (N, 3) float32.
    - A point color array: (N, 3) uint8.
    """
    # Processing blocks. Could be tuned.
    point_cloud = rs.pointcloud()  # type: ignore
    decimate = rs.decimation_filter()  # type: ignore
    decimate.set_option(rs.option.filter_magnitude, 3)  # type: ignore

    # Downsample depth frame.
    depth_frame = decimate.process(depth_frame)

    # Map texture and calculate points from frames. Uses frame intrinsics.
    point_cloud.map_to(color_frame)
    points = point_cloud.calculate(depth_frame)

    # Get color coordinates.
    texture_uv = (
        np.asanyarray(points.get_texture_coordinates())
        .view(np.float32)
        .reshape((-1, 2))
    )
    color_image = np.asanyarray(color_frame.get_data())
    color_h, color_w, _ = color_image.shape

    # Note: for points that aren't in the view of our RGB camera, we currently clamp to
    # the closes available RGB pixel. We could also just remove these points.
    texture_uv = texture_uv.clip(0.0, 1.0)

    # Get positions and colors.
    positions = np.asanyarray(points.get_vertices()).view(np.float32)
    positions = positions.reshape((-1, 3))
    colors = color_image[
        (texture_uv[:, 1] * (color_h - 1.0)).astype(np.int32),
        (texture_uv[:, 0] * (color_w - 1.0)).astype(np.int32),
        :,
    ]
    N = positions.shape[0]

    assert positions.shape == (N, 3)
    assert positions.dtype == np.float32
    assert colors.shape == (N, 3)
    assert colors.dtype == np.uint8

    return positions, colors


def main():
    # Start visualization server.
    viser_server = viser.ViserServer()

    with realsense_pipeline() as pipeline:
        for i in tqdm(range(10000000)):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Compute point cloud from frames.
            positions, colors = point_cloud_arrays_from_frames(depth_frame, color_frame)

            R = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=np.float32,
            )
            positions = positions @ R.T

            # Visualize.
            viser_server.add_point_cloud(
                "/realsense",
                points=positions * 10.0,
                colors=colors,
                point_size=0.1,
            )


if __name__ == "__main__":
    main()
