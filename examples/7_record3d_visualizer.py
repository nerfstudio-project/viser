"""Parse and stream record3d captures. To get the demo data, checkout assets/download_record3d_dance.sh."""

import time
from pathlib import Path
from typing import List, Tuple

import numpy as onp
import numpy.typing as onpt
import tyro
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

import viser


def quat_from_so3(omega: Tuple[float, float, float]) -> onp.ndarray:
    # xyzw => wxyz
    return onp.roll(Rotation.from_rotvec(onp.array(omega)).as_quat(), 1)


def quat_from_mat3(
    mat3: onpt.NDArray[onp.float32],
) -> onp.ndarray:
    # xyzw => wxyz
    return onp.roll(Rotation.from_matrix(mat3).as_quat(), 1)


def main(
    data_path: Path = Path(__file__).parent / "assets/record3d_dance",
    downsample_factor: int = 2,
    max_frames: int = 50,
) -> None:
    server = viser.ViserServer()

    print("Loading frames!")
    loader = viser.extras.Record3dLoader(data_path)
    num_frames = min(max_frames, loader.num_frames())

    # Add playback UI.
    with server.gui_folder("Playback"):
        gui_timestep = server.add_gui_slider(
            "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0
        )
        gui_next_frame = server.add_gui_button("Next Frame")
        gui_prev_frame = server.add_gui_button("Prev Frame")
        gui_playing = server.add_gui_checkbox("Playing", False)
        gui_framerate = server.add_gui_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
        )

    # Frame step buttons.
    @gui_next_frame.on_update
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_update
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        frame_nodes[current_timestep].visible = True
        frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep

    # Load in frames.
    server.add_frame(
        "/frames",
        wxyz=quat_from_so3((onp.pi / 2.0, 0.0, 0.0)),
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: List[viser.SceneNodeHandle] = []
    for i in tqdm(range(num_frames)):
        frame = loader.get_frame(i)
        position, color = frame.get_point_cloud(downsample_factor)
        frame_nodes.append(server.add_frame(f"/frames/t{i}", show_axes=False))
        server.add_point_cloud(
            name=f"/frames/t{i}/pcd", position=position, color=color, point_size=0.01
        )
        server.add_frame(
            f"/frames/t{i}/camera",
            wxyz=quat_from_mat3(frame.T_world_camera[:3, :3]),
            position=frame.T_world_camera[:3, 3],
            axes_length=0.1,
            axes_radius=0.005,
        )
        server.add_camera_frustum(
            f"/frames/t{i}/camera/frustum",
            fov=2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0]),
            aspect=frame.rgb.shape[1] / frame.rgb.shape[0],
            scale=0.15,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
