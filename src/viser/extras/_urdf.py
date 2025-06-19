from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import yourdfpy

import viser

from .. import transforms as tf


class ViserUrdf:
    """Helper for rendering URDFs in Viser. This is a self-contained example
    that uses only basic Viser features. It can be copied and modified if you
    need more fine-grained control.

    To move or control visibility of the entire robot, you can create a
    parent frame that the URDF will be attached to. This is because
    ViserUrdf creates the robot's geometry as children of the specified
    `root_node_name`, but doesn't create the root node itself.

    .. code-block:: python

        import time
        import numpy as np
        import viser
        from viser.extras import ViserUrdf
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        server = viser.ViserServer()

        # Create a parent frame for the robot.
        # ViserUrdf will attach the robot's geometry as children of this frame.
        robot_base = server.scene.add_frame("/robot", show_axes=False)

        # Load a URDF from robot_descriptions package.
        urdf = ViserUrdf(
            server,
            load_robot_description("panda_description"),
            root_node_name="/robot"
        )

        # Move the entire robot by updating the base frame.
        robot_base.position = (1.0, 0.0, 0.5)  # Move to (x=1, y=0, z=0.5).

        # Update joint configuration.
        urdf.update_cfg(np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # Make the robot blink.
        while True:
            robot_base.visible = False
            time.sleep(0.2)
            robot_base.visible = True
            time.sleep(3.0)

    Args:
        target: ViserServer or ClientHandle object to add URDF to.
        urdf_or_path: Either a path to a URDF file or a yourdfpy URDF object.
        scale: Scale factor to apply to resize the URDF.
        root_node_name: Viser scene tree name for the root of the URDF geometry.
        mesh_color_override: Optional color to override the URDF's visual mesh colors.
        collision_mesh_color_override: Optional color to override the URDF's collision mesh colors.
        show_visuals: If true, shows the URDF's visual meshes.
        show_collisions: If true, shows the URDF's collision meshes.
    """

    def __init__(
        self,
        target: viser.ViserServer | viser.ClientHandle,
        urdf_or_path: yourdfpy.URDF | Path,
        scale: float = 1.0,
        root_node_name: str = "/",
        collision_mesh_color_override: tuple[float, float, float] | None = None,
        mesh_color_override: tuple[float, float, float] | None = None,
        show_visuals: bool = True,
        show_collisions: bool = False,
    ) -> None:
        assert root_node_name.startswith("/")
        assert len(root_node_name) == 1 or not root_node_name.endswith("/")

        if isinstance(urdf_or_path, Path):
            urdf = yourdfpy.URDF.load(
                urdf_or_path,
                build_scene_graph=show_visuals,
                build_collision_scene_graph=show_collisions,
                load_meshes=show_visuals,
                load_collision_meshes=show_collisions,
                filename_handler=partial(
                    yourdfpy.filename_handler_magic,
                    dir=urdf_or_path.parent,
                ),
            )
        else:
            urdf = urdf_or_path
        assert isinstance(urdf, yourdfpy.URDF)

        self._target = target
        self._urdf = urdf
        self._scale = scale
        self._root_node_name = root_node_name

        self.collision_root_frame = None
        self.visual_root_frame = None

        self._joint_frames: List[viser.SceneNodeHandle] = []
        self._meshes: List[viser.SceneNodeHandle] = []
        num_joints_to_repeat = 0
        if show_visuals and urdf.scene is not None:
            num_joints_to_repeat += 1
            self._add_joint_frames_and_meshes(
                urdf.scene,
                root_node_name,
                collision_geometry=False,
                mesh_color_override=mesh_color_override,
            )
        if show_collisions and urdf.collision_scene is not None:
            num_joints_to_repeat += 1
            self._add_joint_frames_and_meshes(
                urdf.collision_scene,
                root_node_name,
                collision_geometry=True,
                mesh_color_override=collision_mesh_color_override,
            )

        self._joint_map_values = [*self._urdf.joint_map.values()] * num_joints_to_repeat

    def remove(self) -> None:
        """Remove URDF from scene."""
        # Some of this will be redundant, since children are removed when
        # parents are removed.
        for frame in self._joint_frames:
            frame.remove()
        for mesh in self._meshes:
            mesh.remove()

    def update_cfg(self, configuration: np.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint, frame_handle in zip(
            self._joint_map_values,
            self._joint_frames,
            strict=True,
        ):
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(joint.child, joint.parent)
            frame_handle.wxyz = tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz
            frame_handle.position = T_parent_child[:3, 3] * self._scale

    def get_actuated_joint_limits(
        self,
    ) -> dict[str, tuple[float | None, float | None]]:
        """Returns an ordered mapping from actuated joint names to position limits."""
        out: dict[str, tuple[float | None, float | None]] = {}
        for joint_name, joint in zip(
            self._urdf.actuated_joint_names, self._urdf.actuated_joints
        ):
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            if joint.limit is None:
                out[joint_name] = (-np.pi, np.pi)
            else:
                out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        """Returns a tuple of actuated joint names, in order."""
        return tuple(self._urdf.actuated_joint_names)

    def _add_joint_frames_and_meshes(
        self,
        scene: trimesh.scene.scene.Scene,
        root_node_name: str,
        collision_geometry: bool = False,
        mesh_color_override: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Helper function to add joint frames and meshes to the ViserUrdf object.
        """
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
        root_frame = self._target.scene.add_frame(
            prefixed_root_node_name, show_axes=False
        )
        if collision_geometry:
            self.collision_root_frame = root_frame
        else:
            self.visual_root_frame = root_frame

        # Add coordinate frame for each joint.
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.scene.add_frame(
                    _viser_name_from_frame(
                        scene,
                        joint.child,
                        prefixed_root_node_name,
                    ),
                    show_axes=False,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for link_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = self._urdf.get_transform(
                link_name,
                scene.graph.transforms.parents[link_name],
                collision_geometry=collision_geometry,
            )
            name = _viser_name_from_frame(scene, link_name, prefixed_root_node_name)

            # Scale + transform the mesh. (these will mutate it!)
            #
            # It's important that we use apply_transform() instead of unpacking
            # the rotation/translation terms, since the scene graph transform
            # can also contain scale and reflection terms.
            mesh = mesh.copy()
            mesh.apply_scale(self._scale)
            mesh.apply_transform(T_parent_child)

            if mesh_color_override is None:
                self._meshes.append(self._target.scene.add_mesh_trimesh(name, mesh))
            else:
                self._meshes.append(
                    self._target.scene.add_mesh_simple(
                        name,
                        mesh.vertices,
                        mesh.faces,
                        color=mesh_color_override,
                    )
                )


def _viser_name_from_frame(
    scene: trimesh.scene.scene.Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])
