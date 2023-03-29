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


def so3_from_quat(wxyz: Tuple[float, float, float, float]) -> onp.ndarray:
    return Rotation.from_quat(onp.array(wxyz[1:] + wxyz[:1])).as_rotvec()


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

    with server.gui_folder("View"):
        gui_wireframe = server.add_gui_checkbox("Wireframe", initial_value=False)
        gui_show_controls = server.add_gui_checkbox("Handles", initial_value=True)

        @gui_show_controls.on_update
        def _(_):
            server.set_scene_node_visibility(
                "/reoriented/smpl/controls", gui_show_controls.value()
            )

    with server.gui_folder("Joints"):
        gui_joints: List[viser.GuiHandle[Tuple[float, float, float]]] = []

        # Reset button.
        gui_button = server.add_gui_button("Reset")

        @gui_button.on_update
        def _(_):
            for joint in gui_joints:
                joint.set_value((0.0, 0.0, 0.0))

        # Individual joint controls.
        for i in range(model.NUM_BODY_JOINTS):
            gui_joint = server.add_gui_vector3(
                # +1 to skip the global orientation.
                name=smplx.joint_names.JOINT_NAMES[i + 1],
                initial_value=(0.0, 0.0, 0.0),
                step=0.05,
            )
            gui_joints.append(gui_joint)

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

    for i in range(model.NUM_BODY_JOINTS):
        controls = server.add_transform_controls(
            f"/reoriented/smpl/controls/{i}/controls",
            depth_test=False,
            line_width=2.5,
            scale=0.15,
            disable_axes=True,
            disable_sliders=True,
        )

        def curry_callback(i: int) -> None:
            @controls.on_update
            def _(controls: viser.TransformControlsHandle) -> None:
                axisangle = so3_from_quat(controls.get_state().wxyz)
                gui_joints[i].set_value((axisangle[0], axisangle[1], axisangle[2]))

        curry_callback(i)

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
            wireframe=gui_wireframe.value(),
        )

        joint_positions = output.joints.squeeze(axis=0).detach().cpu().numpy()
        for i in range(model.NUM_BODY_JOINTS):
            server.add_frame(
                f"/reoriented/smpl/controls/{i}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=joint_positions[i + 1],
                show_axes=False,
            )
        time.sleep(0.03)


if __name__ == "__main__":
    tyro.cli(main)
