"""Robot URDF visualizer

Requires yourdfpy and robot_descriptions. Any URDF supported by yourdfpy should work.
- https://github.com/robot-descriptions/robot_descriptions.py
- https://github.com/clemense/yourdfpy

The :class:`viser.extras.ViserUrdf` is a lightweight interface between yourdfpy
and viser. It can also take a path to a local URDF file as input.
"""

from __future__ import annotations

import time

import numpy as onp
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
from viser.extras import ViserUrdf

# A subset of robots available in the robot_descriptions package.
ROBOT_MODEL_LIST = (
    "panda_description",
    "ur10_description",
    "ur3_description",
    "ur5_description",
    "cassie_description",
    "skydio_x2_description",
    "allegro_hand_description",
    "barrett_hand_description",
    "robotiq_2f85_description",
    "atlas_drc_description",
    "atlas_v4_description",
    "draco3_description",
    "g1_description",
    "h1_description",
    "anymal_c_description",
    "go2_description",
    "mini_cheetah_description",
)


class ControllableViserRobot:
    """Helper class that creates a robot + GUI elements for controlling it."""

    def __init__(
        self,
        urdf: yourdfpy.URDF,
        server: viser.ViserServer,
    ):
        loading_modal = server.gui.add_modal("Loading URDF...")
        with loading_modal:
            server.gui.add_markdown("See terminal for progress!")

        # Create a helper for adding URDFs to Viser. This just adds meshes to the scene,
        # helps us set the joint angles, etc.
        self._viser_urdf = ViserUrdf(
            server,
            # This can also be set to a path to a local URDF file.
            urdf_or_path=urdf,
        )
        loading_modal.close()

        # Create sliders in GUI to control robot.
        (
            self._slider_handles,
            self._initial_angles,
        ) = self._create_gui_elements(server, self._viser_urdf)

        self.update_cfg(onp.array([slider.value for slider in self._slider_handles]))

    def update_cfg(self, configuration: onp.ndarray) -> None:
        """Update the configuration, both the GUI handles and the visualized
        robot."""
        assert len(configuration) == len(self._slider_handles)
        for i, slider in enumerate(self._slider_handles):
            slider.value = configuration[i]
        # self._viser_urdf.update_cfg(configuration)

    def reset_joints(self) -> None:
        """Reset all of the joints."""
        for g, initial_angle in zip(self._slider_handles, self._initial_angles):
            g.value = initial_angle

    def remove(self) -> None:
        """Remove the URDF and all GUI elements."""
        self._viser_urdf.remove()
        for slider in self._slider_handles:
            slider.remove()

    @staticmethod
    def _create_gui_elements(
        server: viser.ViserServer, viser_urdf: ViserUrdf
    ) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
        """Crfeate slider for each joint of the robot."""
        slider_handles: list[viser.GuiInputHandle[float]] = []
        initial_angles: list[float] = []
        for joint_name, (
            lower,
            upper,
        ) in viser_urdf.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -onp.pi
            upper = upper if upper is not None else onp.pi
            initial_angle = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
            slider = server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=initial_angle,
            )
            slider.on_update(  # When sliders move, we update the URDF configuration.
                lambda _: viser_urdf.update_cfg(
                    onp.array([slider.value for slider in slider_handles])
                )
            )
            slider_handles.append(slider)
            initial_angles.append(initial_angle)
        return slider_handles, initial_angles


def main() -> None:
    # Start viser server.
    server = viser.ViserServer()

    # We use a dropdown to select the robot.
    robot_model_name = server.gui.add_dropdown("Robot model", ROBOT_MODEL_LIST)
    robot = ControllableViserRobot(
        load_robot_description(robot_model_name.value), server
    )

    # Remove the old robot and add a new one whenever the dropdown changes.
    @robot_model_name.on_update
    def _(_) -> None:
        nonlocal robot
        robot.remove()
        robot = ControllableViserRobot(
            load_robot_description(robot_model_name.value), server
        )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        robot.reset_joints()

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
