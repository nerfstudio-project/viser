# Scene Handles

A handle is created for each object that is added to the scene. These can be
used to read and set state, as well as detect clicks.

When a scene node is added to a server (for example, via
:func:`viser.ViserServer.add_frame()`), state is synchronized between all
connected clients. When a scene node is added to a client (for example, via
:func:`viser.ClientHandle.add_frame()`), state is local to a specific client.

<!-- prettier-ignore-start -->

.. autoclass:: viser.SceneNodeHandle

.. autoclass:: viser.CameraFrustumHandle

.. autoclass:: viser.FrameHandle

.. autoclass:: viser.BatchedAxesHandle

.. autoclass:: viser.GlbHandle

.. autoclass:: viser.Gui3dContainerHandle

.. autoclass:: viser.ImageHandle

.. autoclass:: viser.LabelHandle

.. autoclass:: viser.MeshHandle

.. autoclass:: viser.MeshSkinnedHandle

.. autoclass:: viser.MeshSkinnedBoneHandle

.. autoclass:: viser.PointCloudHandle

.. autoclass:: viser.TransformControlsHandle

<!-- prettier-ignore-end -->
