"""Visualization script for SMPLX.

We need to install the smplx package and download a corresponding set of model
parameters to run this script:
    https://github.com/vchoutas/smplx
"""

import dataclasses
import functools
import time
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as onp
import smplx
import smplx.joint_names
import smplx.lbs
import torch
import tyro
from scipy.spatial.transform import Rotation

import viser


# TODO we should refactor / generalize the transform code that's used across all of the
# examples.
def so3_from_quat(
    wxyz: Union[Tuple[float, float, float, float], onp.ndarray]
) -> onp.ndarray:
    return Rotation.from_quat(onp.roll(onp.asarray(wxyz), -1)).as_rotvec()


def quat_from_mat3(mat3: onp.ndarray) -> onp.ndarray:
    # xyzw => wxyz
    return onp.roll(Rotation.from_matrix(mat3).as_quat(), 1)


def quat_from_so3(*omegas: Tuple[float, float, float]) -> onp.ndarray:
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
    num_betas: int = 10,
    num_expression_coeffs: int = 10,
    ext: Literal["npz", "pkl"] = "npz",
) -> None:
    server = viser.ViserServer()
    model = smplx.create(
        model_path=str(model_path),
        model_type=model_type,
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext,
    )

    # Re-orient the model.
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

    # Main loop. We'll just keep read from the joints, deform the mesh, then sending the
    # updated mesh in a loop. This could be made a lot more efficient.
    gui_elements = make_gui_elements(
        server, num_betas=model.num_betas, num_body_joints=model.NUM_BODY_JOINTS
    )
    while True:
        # Do nothing if no change.
        if not gui_elements.changed:
            time.sleep(0.01)
            continue
        gui_elements.changed = False

        # Get deformed mesh.
        output = model.forward(
            betas=torch.from_numpy(  # type: ignore
                onp.array(
                    [b.value() for b in gui_elements.gui_betas], dtype=onp.float32
                )[None, ...]
            ),
            expression=None,
            return_verts=True,
            body_pose=torch.from_numpy(
                onp.array([j.value() for j in gui_elements.gui_joints[1:]], dtype=onp.float32)[None, ...]  # type: ignore
            ),
            global_orient=torch.from_numpy(onp.array(gui_elements.gui_joints[0].value(), dtype=onp.float32)[None, ...]),  # type: ignore
            return_full_pose=True,
        )
        joint_positions = output.joints.squeeze(axis=0).detach().cpu().numpy()  # type: ignore
        joint_transforms, parents = joint_transforms_and_parents_from_smpl(
            model, output
        )

        # Send mesh to visualizer.
        server.add_mesh(
            "/reoriented/smpl",
            vertices=output.vertices.squeeze(axis=0).detach().cpu().numpy(),  # type: ignore
            faces=model.faces,
            wireframe=gui_elements.gui_wireframe.value(),
        )

        # Update per-joint frames, which are used for transform controls.
        for i in range(model.NUM_BODY_JOINTS + 1):
            server.add_frame(
                f"/reoriented/smpl/joint_{i}",
                wxyz=(1.0, 0.0, 0.0, 0.0)
                if i == 0
                else quat_from_mat3(joint_transforms[parents[i], :3, :3]),
                position=joint_positions[i],
                show_axes=False,
            )


@dataclasses.dataclass
class GuiElements:
    """Structure containing handles for reading from GUI elements."""

    gui_wireframe: viser.GuiHandle[bool]
    gui_betas: List[viser.GuiHandle[float]]
    gui_joints: List[viser.GuiHandle[Tuple[float, float, float]]]

    changed: bool
    """This flag will be flipped to True whenever the mesh needs to be re-generated."""


def make_gui_elements(
    server: viser.ViserServer, num_betas: int, num_body_joints: int
) -> GuiElements:
    """Make GUI elements for interacting with the model."""

    # GUI elements: mesh settings + visibility.
    with server.gui_folder("View"):
        gui_wireframe = server.add_gui_checkbox("Wireframe", initial_value=False)

        @gui_wireframe.on_update
        def _(_):
            out.changed = True

        gui_show_controls = server.add_gui_checkbox("Handles", initial_value=False)

        @gui_show_controls.on_update
        def _(_):
            add_transform_controls(enabled=gui_show_controls.value())

    # GUI elements: shape parameters.
    with server.gui_folder("Shape"):
        gui_reset_shape = server.add_gui_button("Reset Shape")
        gui_random_shape = server.add_gui_button("Random Shape")

        @gui_reset_shape.on_update
        def _(_):
            for beta in gui_betas:
                beta.set_value(0.0)

        @gui_random_shape.on_update
        def _(_):
            for beta in gui_betas:
                beta.set_value(onp.random.normal(loc=0.0, scale=1.0))

        gui_betas = []
        for i in range(num_betas):
            beta = server.add_gui_slider(
                f"beta{i}", min=-5.0, max=5.0, step=0.01, initial_value=0.0
            )
            gui_betas.append(beta)

            @beta.on_update
            def _(_):
                out.changed = True

    # GUI elements: joint angles.
    with server.gui_folder("Joints"):
        # Reset button.
        gui_reset_joints = server.add_gui_button("Reset Joints")
        gui_random_joints = server.add_gui_button("Random Joints")

        @gui_reset_joints.on_update
        def _(_):
            for joint in gui_joints:
                joint.set_value((0.0, 0.0, 0.0))
                sync_transform_controls()

        @gui_random_joints.on_update
        def _(_):
            for joint in gui_joints:
                # It's hard to uniformly sample orientations directly in so(3), so we
                # first sample on S^3 and then convert.
                quat = onp.random.normal(loc=0.0, scale=1.0, size=(4,))
                quat /= onp.linalg.norm(quat)
                joint.set_value(so3_from_quat(quat))
                sync_transform_controls()

        gui_joints: List[viser.GuiHandle[Tuple[float, float, float]]] = []
        for i in range(num_body_joints + 1):
            gui_joint = server.add_gui_vector3(
                name=smplx.joint_names.JOINT_NAMES[i],
                initial_value=(0.0, 0.0, 0.0),
                step=0.05,
            )
            gui_joints.append(gui_joint)

            @gui_joint.on_update
            def _(_):
                sync_transform_controls()
                out.changed = True

    # Transform control gizmos on joints.
    transform_controls: List[viser.TransformControlsHandle] = []

    def add_transform_controls(enabled: bool) -> List[viser.TransformControlsHandle]:
        for i in range(num_body_joints + 1):
            controls = server.add_transform_controls(
                f"/reoriented/smpl/joint_{i}/controls",
                depth_test=False,
                line_width=3.5 if i == 0 else 2.0,
                scale=0.2 if i == 0 else 0.1,
                disable_axes=True,
                disable_sliders=True,
                disable_rotations=not enabled,
            )
            transform_controls.append(controls)

            def curry_callback(i: int) -> None:
                @controls.on_update
                def _(controls: viser.TransformControlsHandle) -> None:
                    axisangle = so3_from_quat(controls.get_state().wxyz)
                    gui_joints[i].set_value((axisangle[0], axisangle[1], axisangle[2]))

            curry_callback(i)

        return transform_controls

    def sync_transform_controls() -> None:
        """Sync transform controls when a joint angle changes."""
        for t, j in zip(transform_controls, gui_joints):
            t.set_state(quat_from_so3(j.value()), t.get_state().position)

    add_transform_controls(enabled=False)

    out = GuiElements(gui_wireframe, gui_betas, gui_joints, changed=True)
    return out


def joint_transforms_and_parents_from_smpl(model, output):
    """Hack at SMPL internals to get coordinate frames corresponding to each joint."""
    v_shaped = model.v_template + smplx.lbs.blend_shapes(  # type: ignore
        model.betas, model.shapedirs  # type: ignore
    )
    J = smplx.lbs.vertices2joints(model.J_regressor, v_shaped)  # type: ignore
    rot_mats = smplx.lbs.batch_rodrigues(output.full_pose.view(-1, 3)).view(  # type: ignore
        [1, -1, 3, 3]
    )
    J_posed, A = smplx.lbs.batch_rigid_transform(rot_mats, J, model.parents)  # type: ignore
    transforms = A.detach().cpu().numpy().squeeze(axis=0)  # type: ignore
    parents = model.parents.detach().cpu().numpy()  # type: ignore
    return transforms, parents


if __name__ == "__main__":
    tyro.cli(main, description=__doc__)
