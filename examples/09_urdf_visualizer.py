"""URDF visualizer

Requires yourdfpy and URDF. Any URDF supported by yourdfpy should work.

Examples:
- https://github.com/OrebroUniversity/yumi/blob/master/yumi_description/urdf/yumi.urdf
- https://github.com/ankurhanda/robot-assets
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as onp
import tyro

import viser
from viser.extras import ViserUrdf


def main(urdf_path: Path) -> None:
    server = viser.ViserServer()

    # Create a helper for adding URDFs to Viser. This just adds meshes to the scene,
    # helps us set the joint angles, etc.
    urdf = ViserUrdf(server, urdf_path)

    # Create joint angle sliders.
    gui_joints: List[viser.GuiHandle[float]] = []
    with server.add_gui_folder("Joints"):
        server.add_gui_button("Reset")

        gui_joints.append(slider)
        initial_angles.append(initial_angle)

    # Create joint reset button.
    reset_button = server.add_gui_button("Reset")

    @reset_button.on_click
    def _(_):
        for g, initial_angle in zip(gui_joints, initial_angles):
            g.value = initial_angle

    # Apply initial joint angles.
    urdf.update_cfg(onp.array([gui.value for gui in gui_joints]))

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
