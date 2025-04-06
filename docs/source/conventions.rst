Frame Conventions
=================

In this note, we describe the coordinate frame conventions used in ``viser``.

Scene tree naming
-----------------

Each object that we add to the scene in viser is instantiated as a node in a
scene tree. The structure of this tree is determined by the names assigned to
the nodes.

If we add a coordinate frame called ``/base_link/shoulder/wrist``, it signifies
three nodes: the ``wrist`` is a child of the ``shoulder`` which is a child of the
``base_link``.

If we set the transformation of a given node like ``/base_link/shoulder``, both
it and its child ``/base_link/shoulder/wrist`` will move. Its parent,
``/base_link``, will be unaffected.

Poses
-----

Poses in ``viser`` are defined using a pair of fields:

- ``wxyz``, a unit quaternion orientation term. This should always be 4D.
- ``position``, a translation term. This should always be 3D.

These correspond to a transformation from coordinates in the local frame to the
parent frame:

.. math::

   p_\mathrm{parent} = \begin{bmatrix} R & t \end{bmatrix}\begin{bmatrix}p_\mathrm{local} \\ 1\end{bmatrix}

where ``wxyz`` is the quaternion form of the :math:`\mathrm{SO}(3)` matrix
:math:`R` and ``position`` is the :math:`\mathbb{R}^3` translation term
:math:`t`.

World coordinates
-----------------

In the world coordinate space, +Z points upward by default. This can be
overridden with :func:`viser.SceneApi.set_up_direction()`.

Cameras
-------

In ``viser``, all camera parameters exposed to the Python API use the
COLMAP/OpenCV convention:

- Forward: +Z
- Up: -Y
- Right: +X

Confusingly, this is different from Nerfstudio, which adopts the OpenGL/Blender
convention:

- Forward: -Z
- Up: +Y
- Right: +X

Conversion between the two is a simple 180 degree rotation around the local X-axis.
