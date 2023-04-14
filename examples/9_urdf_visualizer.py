"""Visualize a URDF file + its joints.

Requires yourdfpy and URDF. I've been using:
    https://github.com/OrebroUniversity/yumi/blob/master/yumi_description/urdf/yumi.urdf
"""
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import numpy as onp
import trimesh
import trimesh.transformations as tf
import tyro
import yourdfpy
from scipy.spatial.transform import Rotation

import viser


def so3_from_quat(
    wxyz: Union[Tuple[float, float, float, float], onp.ndarray]
) -> onp.ndarray:
    return Rotation.from_quat(onp.roll(onp.asarray(wxyz), -1)).as_rotvec()


def main(urdf_path: Path):
    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
    )
    server = viser.ViserServer()

    def frame_name_with_parents(frame_name: str):
        frames = []
        while frame_name != "world":
            frames.append(frame_name)
            frame_name = urdf.scene.graph.transforms.parents[frame_name]
        return "/" + "/".join(frames[::-1])

    for frame_name, value in urdf.scene.geometry.items():
        assert isinstance(value, trimesh.Trimesh)
        server.add_mesh(
            frame_name_with_parents(frame_name) + "/mesh",
            vertices=value.vertices,
            faces=value.faces,
            color=(150, 150, 150),
        )

    gui_joints: List[viser.GuiHandle[float]] = []
    with server.gui_folder("Joints"):
        button = server.add_gui_button("Reset")

        @button.on_update
        def _(_):
            for g in gui_joints:
                g.set_value(0.0)

        def update_frames():
            urdf.update_cfg(onp.array([gui.get_value() for gui in gui_joints]))
            for joint in urdf.joint_map.values():
                assert isinstance(joint, yourdfpy.Joint)
                T_parent_child = urdf.get_transform(joint.child, joint.parent)
                server.add_frame(
                    frame_name_with_parents(joint.child),
                    wxyz=tf.quaternion_from_matrix(T_parent_child[:3, :3]),
                    position=T_parent_child[:3, 3],
                    show_axes=False,
                )

        for joint_name, joint in urdf.joint_map.items():
            assert isinstance(joint, yourdfpy.Joint)
            slider = server.add_gui_slider(
                name=joint_name,
                min=(
                    joint.limit.lower
                    if joint.limit is not None and joint.limit.lower is not None
                    else -onp.pi
                ),
                max=(
                    joint.limit.upper
                    if joint.limit is not None and joint.limit.upper is not None
                    else onp.pi
                ),
                step=1e-3,
                initial_value=0.0,
            )
            if joint.limit is None:
                slider.set_hidden(True)

            @slider.on_update
            def _(_):
                update_frames()

            gui_joints.append(slider)

    update_frames()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
