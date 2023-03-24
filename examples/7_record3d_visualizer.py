"""Parse and stream record3d captures. To get the demo data, checkout assets/download_record3d_dance.sh."""

import time
from pathlib import Path
from typing import Tuple

import numpy as onp
import numpy.typing as onpt
import tyro
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

import viser


def quat_from_so3(
    omega: Tuple[float, float, float]
) -> Tuple[float, float, float, float]:
    return tuple(onp.roll(Rotation.from_rotvec(onp.array(omega)).as_quat(), 1).tolist())


def quat_from_mat3(
    mat3: onpt.NDArray[onp.float32],
) -> Tuple[float, float, float, float]:
    return tuple(onp.roll(Rotation.from_matrix(mat3).as_quat(), 1).tolist())


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
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.add_gui_button("Next Frame", disabled=True)
        gui_prev_frame = server.add_gui_button("Prev Frame", disabled=True)
        gui_playing = server.add_gui_checkbox(
            "Playing",
            False,
            disabled=True,
        )
        gui_framerate = server.add_gui_slider(
            "FPS",
            min=1,
            max=60,
            step=0.1,
            initial_value=loader.fps,
            disabled=True,
        )

    # Frame step buttons.
    @gui_next_frame.on_update
    def _(_) -> None:
        gui_timestep.set_value((gui_timestep.value() + 1) % num_frames)

    @gui_prev_frame.on_update
    def _(_) -> None:
        gui_timestep.set_value((gui_timestep.value() - 1) % num_frames)

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.set_disabled(gui_playing.value())
        gui_next_frame.set_disabled(gui_playing.value())
        gui_prev_frame.set_disabled(gui_playing.value())

    prev_timestep = gui_timestep.value()

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value()
        server.set_scene_node_visibility(f"/frames/t{current_timestep}", True)
        server.set_scene_node_visibility(f"/frames/t{prev_timestep}", False)
        prev_timestep = current_timestep

    # Load in frames.
    server.add_frame(
        "/frames",
        wxyz=quat_from_so3((onp.pi / 2.0, 0.0, 0.0)),
        position=(0, 0, 0),
        show_axes=False,
    )
    for i in tqdm(range(num_frames)):
        frame = loader.get_frame(i)
        position, color = frame.get_point_cloud(downsample_factor)
        server.add_point_cloud(
            name=f"/frames/t{i}/pcd", position=position, color=color, point_size=0.01
        )
        server.add_frame(
            f"/frames/t{i}/camera",
            wxyz=quat_from_mat3(frame.T_world_camera[:3, :3]),
            position=tuple(frame.T_world_camera[:3, 3].tolist()),
            scale=0.1,
        )
        server.add_camera_frustum(
            f"/frames/t{i}/camera/frustum",
            fov=2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0]),
            aspect=frame.rgb.shape[1] / frame.rgb.shape[0],
            scale=0.1,
        )

    # Remove loading progress indicator.
    server.set_scene_node_visibility(f"/axes", False)

    # Undisable UI after the frames are loaded.
    gui_timestep.set_disabled(False)
    gui_next_frame.set_disabled(False)
    gui_prev_frame.set_disabled(False)
    gui_playing.set_disabled(False)
    gui_framerate.set_disabled(False)

    # Hide all but the current frame.
    for i in range(num_frames):
        server.set_scene_node_visibility(f"/frames/t{i}", i == gui_timestep.value())

    # Playback update loop.
    prev_timestep = gui_timestep.value()
    while True:
        if gui_playing.value():
            gui_timestep.set_value((gui_timestep.value() + 1) % num_frames)

        time.sleep(1.0 / gui_framerate.value())


if __name__ == "__main__":
    tyro.cli(main)
