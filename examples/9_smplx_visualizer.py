import functools
import time
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as onp
import smplx
import smplx.joint_names
import smplx.lbs
import smplx.utils
import torch
import tyro
from scipy.spatial.transform import Rotation

import viser


def quat_from_so3(*omegas: Tuple[float, float, float]) -> onp.ndarray:
    """TODO: we should refactor / generalize this."""
    # xyzw => wxyz
    return onp.roll(
        functools.reduce(
            Rotation.__mul__,
            [Rotation.from_rotvec(onp.array(omega)) for omega in omegas],
        ).as_quat(),
        1,
    )


def main(
    model_path: Path,
    model_type: Literal["smpl", "smplh", "smplx", "mano"] = "smplx",
    gender: Literal["male", "female", "neutral"] = "neutral",
    use_face_contour: bool = False,
    num_betas: int = 10,
    num_expression_coeffs: int = 10,
    ext: Literal["npz", "pkl"] = "npz",
) -> None:
    server = viser.ViserServer()
    model = smplx.create(
        model_path=str(model_path),
        model_type=model_type,
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext,
    )

    with server.gui_folder("Joints"):
        gui_joints = []

        # Individual joint controls.
        for i in range(model.NUM_BODY_JOINTS):
            gui_joint = server.add_gui_vector3(
                # +1 to skip the global orientation.
                name=smplx.joint_names.JOINT_NAMES[i + 1],
                initial_value=(0.0, 0.0, 0.0),
                step=0.05,
            )
            gui_joints.append(gui_joint)

        # Reset button.
        gui_button = server.add_gui_button("Reset")

        @gui_button.on_update
        def _(_):
            for joint in gui_joints:
                joint.set_value((0.0, 0.0, 0.0))

    server.add_frame(
        "/reoriented",
        wxyz=quat_from_so3(
            (0.0, 0.0, onp.pi),
            (onp.pi / 2.0, 0.0, 0.0),
        ),
        position=onp.zeros(3),
        show_axes=False,
    )
    server.set_scene_node_visibility("/WorldAxes", False)

    while True:
        output = model.forward(
            betas=None,
            expression=None,
            return_verts=True,
            body_pose=torch.from_numpy(
                onp.array([j.value() for j in gui_joints], dtype=onp.float32)[None, ...]
            ),
        )
        server.add_mesh(
            "/reoriented/smpl",
            vertices=output.vertices.squeeze(axis=0).detach().cpu().numpy(),
            faces=model.faces,
        )

        for i in range(model.NUM_BODY_JOINTS):
            server.add_frame(
                f"/reoriented/smpl/{i}",
                wxyz=quat_from_so3(gui_joints[i].value()),
                position=output.joints.squeeze(axis=0)[i + 1].detach().cpu().numpy(),
                axes_radius=0.02,
                axes_length=0.2,
            )

        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
