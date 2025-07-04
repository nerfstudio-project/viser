.. This file automatically includes all scene handles exported from viser.__init__.py
.. Filtering is done by filter_handles_by_page() in conf.py which checks obj.__module__
.. to include only handles from viser._scene_handles on this page

Scene Handles
=============

A handle is created for each object that is added to the scene. These can be
used to read and set state, as well as detect clicks.

When a scene node is added to a server (for example, via
:func:`viser.ViserServer.add_frame()`), state is synchronized between all
connected clients. When a scene node is added to a client (for example, via
:func:`viser.ClientHandle.add_frame()`), state is local to a specific client.

The most common attributes to read and write here are
:attr:`viser.SceneNodeHandle.wxyz` and :attr:`viser.SceneNodeHandle.position`.
Each node type also has type-specific attributes that we can read and write.
Many of these are lower-level than their equivalent arguments in factory
methods like :func:`viser.ViserServer.add_frame()` or
:func:`viser.ViserServer.add_image()`.

.. automodule:: viser
   :members:
   :undoc-members:
   :inherited-members: