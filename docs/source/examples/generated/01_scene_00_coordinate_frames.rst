Visualize 3D coordinate frames and transformations.
===================================================

In this basic example, we visualize a set of coordinate frames.

**Source:** ``examples_dev/01_scene/00_coordinate_frames.py``

.. figure:: ../_static/examples/01_scene_00_coordinate_frames.png
   :width: 100%
   :alt: Visualize 3D coordinate frames and transformations.

Code
----

.. code-block:: python
   :linenos:

   """Visualize 3D coordinate frames and transformations.
   
   In this basic example, we visualize a set of coordinate frames.
   
   Naming for all scene nodes are hierarchical; /tree/branch, for example, is defined
   relative to /tree.
   """
   
   import random
   import time
   
   import viser
   
   server = viser.ViserServer()
   
   while True:
       # Add some coordinate frames to the scene. These will be visualized in the viewer.
       server.scene.add_frame(
           "/tree",
           wxyz=(1.0, 0.0, 0.0, 0.0),
           position=(random.random() * 2.0, 2.0, 0.2),
       )
       server.scene.add_frame(
           "/tree/branch",
           wxyz=(1.0, 0.0, 0.0, 0.0),
           position=(random.random() * 2.0, 2.0, 0.2),
       )
       leaf = server.scene.add_frame(
           "/tree/branch/leaf",
           wxyz=(1.0, 0.0, 0.0, 0.0),
           position=(random.random() * 2.0, 2.0, 0.2),
       )
       time.sleep(5.0)
   
       # Remove the leaf node from the scene.
       leaf.remove()
       time.sleep(0.5)
   
