"""URDF visualizer

Requires yourdfpy and URDF. Any URDF supported by yourdfpy should work.

Examples:
- https://github.com/OrebroUniversity/yumi/blob/master/yumi_description/urdf/yumi.urdf
- https://github.com/ankurhanda/robot-assets
"""
import time
from functools import partial
from pathlib import Path
from typing import List

import numpy as onp
import trimesh
import tyro
import yourdfpy

import viser
import viser.transforms as tf


def main(urdf_path: Path) -> None:
    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
    )
    server = viser.ViserServer()

    def frame_name_with_parents(frame_name: str) -> str:
        frames = []
        while frame_name != urdf.scene.graph.base_frame:
            frames.append(frame_name)
            frame_name = urdf.scene.graph.transforms.parents[frame_name]
        return "/" + "/".join(frames[::-1])

    for frame_name, mesh in urdf.scene.geometry.items():
        assert isinstance(mesh, trimesh.Trimesh)
        T_parent_child = urdf.get_transform(
            frame_name, urdf.scene.graph.transforms.parents[frame_name]
        )
        server.add_mesh_trimesh(
            frame_name_with_parents(frame_name),
            mesh,
            wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
            position=T_parent_child[:3, 3],
        )

    gui_joints: List[viser.GuiHandle[float]] = []
    with server.gui_folder("Joints"):
        button = server.add_gui_button("Reset")

        @button.on_click
        def _(_):
            for g in gui_joints:
                g.value = 0.0

        def update_frames():
            urdf.update_cfg(onp.array([gui.value for gui in gui_joints]))
            for joint in urdf.joint_map.values():
                assert isinstance(joint, yourdfpy.Joint)
                T_parent_child = urdf.get_transform(joint.child, joint.parent)
                server.add_frame(
                    frame_name_with_parents(joint.child),
                    wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                    position=T_parent_child[:3, 3],
                    show_axes=False,
                )

        for joint_name, joint in urdf.joint_map.items():
            assert isinstance(joint, yourdfpy.Joint)

            min = (
                joint.limit.lower
                if joint.limit is not None and joint.limit.lower is not None
                else -onp.pi
            )
            max = (
                joint.limit.upper
                if joint.limit is not None and joint.limit.upper is not None
                else onp.pi
            )
            slider = server.add_gui_slider(
                name=joint_name,
                min=min,
                max=max,
                step=1e-3,
                initial_value=0.0 if min < 0 and max > 0 else (min + max) / 2.0,
            )
            if joint.limit is None:
                slider.visible = False

            @slider.on_update
            def _(_):
                update_frames()

            gui_joints.append(slider)

    update_frames()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
