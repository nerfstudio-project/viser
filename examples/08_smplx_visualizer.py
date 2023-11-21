# mypy: disable-error-code="assignment"
#
# Asymmetric properties are supported in Pyright, but not yet in mypy.
# - https://github.com/python/mypy/issues/3004
# - https://github.com/python/mypy/pull/11643
"""SMPL-X visualizer

We need to install the smplx package and download a corresponding set of model
parameters to run this script:
- https://github.com/vchoutas/smplx
"""

import dataclasses
import time
from pathlib import Path
from typing import List, Tuple

import numpy as onp
import smplx
import smplx.joint_names
import smplx.lbs
import torch
import tyro
from typing_extensions import Literal

import viser
import viser.transforms as tf


def main(
    model_path: Path,
    model_type: Literal["smpl", "smplh", "smplx", "mano"] = "smplx",
    gender: Literal["male", "female", "neutral"] = "neutral",
    num_betas: int = 10,
    num_expression_coeffs: int = 10,
    ext: Literal["npz", "pkl"] = "npz",
    share: bool = False,
) -> None:
    server = viser.ViserServer(share=share)
    server.configure_theme(control_layout="collapsible")
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
        wxyz=(
            tf.SO3.exp(onp.array((0.0, 0.0, onp.pi)))
            @ tf.SO3.exp(onp.array((onp.pi / 2.0, 0.0, 0.0)))
        ).wxyz,
        position=onp.zeros(3),
        show_axes=False,
    )

    # Main loop. We'll just keep read from the joints, deform the mesh, then sending the
    # updated mesh in a loop. This could be made a lot more efficient.
    gui_elements = make_gui_elements(
        server, num_betas=model.num_betas, num_body_joints=int(model.NUM_BODY_JOINTS)
    )
    while True:
        # Do nothing if no change.
        if not gui_elements.changed:
            time.sleep(0.01)
            continue
        gui_elements.changed = False

        full_pose = torch.from_numpy(
            onp.array([j.value for j in gui_elements.gui_joints[1:]], dtype=onp.float32)[None, ...]  # type: ignore
        )

        # Get deformed mesh.
        output = model.forward(
            betas=torch.from_numpy(  # type: ignore
                onp.array([b.value for b in gui_elements.gui_betas], dtype=onp.float32)[
                    None, ...
                ]
            ),
            expression=None,
            return_verts=True,
            body_pose=full_pose[:, : model.NUM_BODY_JOINTS],  # type: ignore
            global_orient=torch.from_numpy(onp.array(gui_elements.gui_joints[0].value, dtype=onp.float32)[None, ...]),  # type: ignore
            return_full_pose=True,
        )
        joint_positions = output.joints.squeeze(axis=0).detach().cpu().numpy()  # type: ignore
        joint_transforms, parents = joint_transforms_and_parents_from_smpl(
            model, output
        )

        # Send mesh to visualizer.
        server.add_mesh_simple(
            "/reoriented/smpl",
            vertices=output.vertices.squeeze(axis=0).detach().cpu().numpy(),  # type: ignore
            faces=model.faces,
            wireframe=gui_elements.gui_wireframe.value,
            color=gui_elements.gui_rgb.value,
        )

        # Update per-joint frames, which are used for transform controls.
        for i in range(model.NUM_BODY_JOINTS + 1):
            R = joint_transforms[parents[i], :3, :3]
            server.add_frame(
                f"/reoriented/smpl/joint_{i}",
                wxyz=((1.0, 0.0, 0.0, 0.0) if i == 0 else tf.SO3.from_matrix(R).wxyz),
                position=joint_positions[i],
                show_axes=False,
            )


@dataclasses.dataclass
class GuiElements:
    """Structure containing handles for reading from GUI elements."""

    gui_rgb: viser.GuiInputHandle[Tuple[int, int, int]]
    gui_wireframe: viser.GuiInputHandle[bool]
    gui_betas: List[viser.GuiInputHandle[float]]
    gui_joints: List[viser.GuiInputHandle[Tuple[float, float, float]]]

    changed: bool
    """This flag will be flipped to True whenever the mesh needs to be re-generated."""


def make_gui_elements(
    server: viser.ViserServer, num_betas: int, num_body_joints: int
) -> GuiElements:
    """Make GUI elements for interacting with the model."""

    tab_group = server.add_gui_tab_group()

    # GUI elements: mesh settings + visibility.
    with tab_group.add_tab("View", viser.Icon.VIEWFINDER):
        gui_rgb = server.add_gui_rgb("Color", initial_value=(90, 200, 255))
        gui_rgb_text = server.add_gui_text("Color", "")
        gui_wireframe = server.add_gui_checkbox("Wireframe", initial_value=False)
        gui_show_controls = server.add_gui_checkbox("Handles", initial_value=False)

        @gui_rgb.on_update
        def _(_):
            out.changed = True

        @gui_wireframe.on_update
        def _(_):
            out.changed = True

        @gui_show_controls.on_update
        def _(_):
            add_transform_controls(enabled=gui_show_controls.value)

    # GUI elements: shape parameters.
    with tab_group.add_tab("Shape", viser.Icon.BOX):
        gui_reset_shape = server.add_gui_button("Reset Shape")
        gui_random_shape = server.add_gui_button("Random Shape")

        @gui_reset_shape.on_click
        def _(_):
            for beta in gui_betas:
                beta.value = 0.0

        @gui_random_shape.on_click
        def _(_):
            for beta in gui_betas:
                beta.value = onp.random.normal(loc=0.0, scale=1.0)

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
    with tab_group.add_tab("Joints", viser.Icon.ANGLE):
        gui_reset_joints = server.add_gui_button("Reset Joints")
        gui_random_joints = server.add_gui_button("Random Joints")

        @gui_reset_joints.on_click
        def _(_):
            for joint in gui_joints:
                joint.value = (0.0, 0.0, 0.0)
                sync_transform_controls()

        @gui_random_joints.on_click
        def _(_):
            for joint in gui_joints:
                # It's hard to uniformly sample orientations directly in so(3), so we
                # first sample on S^3 and then convert.
                quat = onp.random.normal(loc=0.0, scale=1.0, size=(4,))
                quat /= onp.linalg.norm(quat)

                # xyzw => wxyz => so(3)
                joint.value = tf.SO3(wxyz=quat).log()
                sync_transform_controls()

        gui_joints: List[viser.GuiInputHandle[Tuple[float, float, float]]] = []
        for i in range(num_body_joints + 1):
            gui_joint = server.add_gui_vector3(
                label=smplx.joint_names.JOINT_NAMES[i],
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
        for i in range(1 + num_body_joints):
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
                    axisangle = tf.SO3(controls.wxyz).log()
                    gui_joints[i].value = (axisangle[0], axisangle[1], axisangle[2])

            curry_callback(i)

        return transform_controls

    def sync_transform_controls() -> None:
        """Sync transform controls when a joint angle changes."""
        for t, j in zip(transform_controls, gui_joints):
            t.wxyz = tf.SO3.exp(onp.array(j.value)).wxyz

    add_transform_controls(enabled=False)

    out = GuiElements(gui_rgb, gui_wireframe, gui_betas, gui_joints, changed=True)
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
