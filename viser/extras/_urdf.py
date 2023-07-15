from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as onp
import trimesh
import yourdfpy

import viser

from .. import transforms as tf


class ViserUrdf:
    """Helper for rendering URDFs in Viser."""

    def __init__(
        self, target: Union[viser.ViserServer, viser.ClientHandle], urdf_path: Path
    ) -> None:
        self._target = target
        self._urdf: yourdfpy.URDF = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=urdf_path.parent
            ),
        )
        _add_urdf_meshes(target, self._urdf)

    def update_cfg(self, configuration: onp.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(joint.child, joint.parent)
            self._target.add_frame(
                _viser_name_from_frame(self._urdf, joint.child),
                wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                position=T_parent_child[:3, 3],
                show_axes=False,
            )

    def get_joint_limits(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Returns an ordered mapping from joint names to position limits."""
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for joint_name, joint in self._urdf.joint_map.items():
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            if joint.limit is None:
                # Dummy (?) joint.
                continue
            out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out


def _add_urdf_meshes(
    target: Union[viser.ViserServer, viser.ClientHandle],
    urdf: yourdfpy.URDF,
) -> None:
    """Add meshes for a URDF file to viser."""
    for link_name, mesh in urdf.scene.geometry.items():
        assert isinstance(mesh, trimesh.Trimesh)
        T_parent_child = urdf.get_transform(
            link_name, urdf.scene.graph.transforms.parents[link_name]
        )
        target.add_mesh_trimesh(
            _viser_name_from_frame(urdf, link_name),
            mesh,
            wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
            position=T_parent_child[:3, 3],
        )


def _viser_name_from_frame(urdf: yourdfpy.URDF, frame_name: str) -> str:
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


    this would map a name like "elbow" to "/base_link/shoulder/elbow".
    """
    frames = []
    while frame_name != urdf.scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = urdf.scene.graph.transforms.parents[frame_name]
    return "/" + "/".join(frames[::-1])
