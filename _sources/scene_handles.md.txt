# Scene Handles

A handle is created for each object that is added to the scene. These can be
used to read and set state, as well as detect clicks.

When a scene node is added to a server (for example, via
:func:`ViserServer.add_frame()`), state is synchronized between all connected
clients. When a scene node is added to a client (for example, via
:func:`ClientHandle.add_frame()`), state is local to a specific client.

<!-- prettier-ignore-start -->

.. autoapiclass:: viser.SceneNodeHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.CameraFrustumHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.FrameHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.Gui3dContainerHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.ImageHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.LabelHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.MeshHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.PointCloudHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.TransformControlsHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.ClickEvent
   :members:
   :undoc-members:
   :inherited-members:

<!-- prettier-ignore-end -->
