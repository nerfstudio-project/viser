Scene API
=========

The :class:`~viser.SceneApi` provides methods for adding and manipulating 3D objects in your visualization. Access it via ``server.scene``.

.. note::
   All scene objects are organized in a hierarchical **scene graph**. Objects can be grouped using the ``/`` separator in names (e.g., ``"robot/arm/joint1"``).

Basic 3D Objects
-----------------

**Primitive Shapes:**

.. code-block:: python

   import viser
   
   server = viser.ViserServer()
   
   # Basic shapes
   server.scene.add_icosphere("sphere", radius=0.5, color=(255, 0, 0))
   server.scene.add_box("box", dimensions=(1, 1, 1), color=(0, 255, 0))
   server.scene.add_cylinder("cylinder", height=2.0, radius=0.3, color=(0, 0, 255))
   
   # Coordinate frame for reference
   server.scene.add_frame("world_frame", axes_length=1.0, axes_radius=0.02)

**Point Clouds:**

.. code-block:: python

   import numpy as np
   
   # Random point cloud
   points = np.random.randn(1000, 3)
   colors = np.random.randint(0, 255, size=(1000, 3))
   
   server.scene.add_point_cloud(
       "random_points",
       points=points,
       colors=colors,
       point_size=0.02
   )

**Lines and Paths:**

.. code-block:: python

   # Draw a trajectory
   trajectory_points = np.array([
       [0, 0, 0],
       [1, 0, 0], 
       [1, 1, 0],
       [0, 1, 0],
       [0, 0, 1]
   ])
   
   server.scene.add_line_segments(
       "path",
       points=trajectory_points[:-1],
       points_to=trajectory_points[1:],
       color=(255, 255, 0),
       line_width=3.0
   )

Complex Geometry
----------------

**Meshes from Files:**

.. code-block:: python

   import trimesh
   
   # Load and display a mesh
   mesh = trimesh.load_mesh("path/to/model.obj")
   server.scene.add_mesh_simple(
       "loaded_mesh",
       vertices=mesh.vertices,
       faces=mesh.faces,
       color=(150, 150, 150)
   )

**Custom Meshes:**

.. code-block:: python

   # Create a simple triangle
   vertices = np.array([
       [0, 0, 0],
       [1, 0, 0],
       [0.5, 1, 0]
   ])
   faces = np.array([[0, 1, 2]])
   
   server.scene.add_mesh_simple(
       "triangle",
       vertices=vertices,
       faces=faces,
       color=(255, 100, 100)
   )

**Textured Meshes:**

.. code-block:: python

   import imageio.v3 as iio
   
   # Mesh with texture
   texture_image = iio.imread("texture.png")
   server.scene.add_mesh_simple(
       "textured_mesh",
       vertices=mesh.vertices,
       faces=mesh.faces,
       vertex_colors=vertex_colors,  # Per-vertex colors
       # texture=texture_image  # Coming soon!
   )

Scene Organization
------------------

**Hierarchical Names:**

.. code-block:: python

   # Create a robot hierarchy
   server.scene.add_box("robot/base", dimensions=(0.3, 0.3, 0.1))
   server.scene.add_cylinder("robot/base/arm", height=0.5, radius=0.05)
   server.scene.add_icosphere("robot/base/arm/gripper", radius=0.03)
   
   # Hide entire robot
   server.scene["robot"].visible = False
   
   # Remove specific part
   server.scene["robot/base/arm"].remove()

**Coordinate Frames:**

.. code-block:: python

   # Camera poses
   camera_positions = [(0, 0, 2), (1, 0, 2), (0, 1, 2)]
   
   for i, pos in enumerate(camera_positions):
       server.scene.add_frame(
           f"camera_{i}",
           axes_length=0.2,
           axes_radius=0.01,
           position=pos
       )

**Grouping and Transforms:**

.. code-block:: python

   import viser.transforms as tf
   import numpy as np
   
   # Create objects in a group
   group_objects = [
       ("sphere", {"radius": 0.2, "color": (255, 0, 0)}),
       ("box", {"dimensions": (0.2, 0.2, 0.2), "color": (0, 255, 0)}),
       ("cylinder", {"height": 0.4, "radius": 0.1, "color": (0, 0, 255)})
   ]
   
   for i, (shape, params) in enumerate(group_objects):
       name = f"group/object_{i}"
       position = (i * 0.5, 0, 0)
       
       if shape == "sphere":
           server.scene.add_icosphere(name, position=position, **params)
       elif shape == "box":
           server.scene.add_box(name, position=position, **params)
       elif shape == "cylinder":
           server.scene.add_cylinder(name, position=position, **params)
   
   # Rotate entire group
   group_rotation = tf.SO3.from_z_radians(np.pi / 4)
   server.scene["group"].wxyz = group_rotation.wxyz

Real-time Updates
-----------------

**Modifying Object Properties:**

.. code-block:: python

   # Create a sphere we can modify
   sphere_handle = server.scene.add_icosphere("moving_sphere", radius=0.3, color=(255, 0, 0))
   
   import time
   import numpy as np
   
   # Animation loop
   t = 0
   while True:
       # Update position
       x = np.cos(t)
       y = np.sin(t)
       z = 0.2 * np.sin(2 * t)
       sphere_handle.position = (x, y, z)
       
       # Update color
       r = int(127 + 127 * np.cos(t))
       g = int(127 + 127 * np.sin(t))
       sphere_handle.color = (r, g, 100)
       
       t += 0.1
       time.sleep(0.05)

**Dynamic Scene Updates:**

.. code-block:: python

   # Add/remove objects dynamically
   object_count = 0
   
   def add_random_object():
       global object_count
       pos = np.random.randn(3)
       color = tuple(np.random.randint(0, 255, 3))
       
       server.scene.add_icosphere(
           f"dynamic_{object_count}",
           radius=0.1,
           position=pos,
           color=color
       )
       object_count += 1
   
   def clear_dynamic_objects():
       for i in range(object_count):
           try:
               server.scene[f"dynamic_{i}"].remove()
           except KeyError:
               pass  # Already removed

Performance Optimization
------------------------

**Batched Meshes:**

.. code-block:: python

   # For many similar objects, use batching
   positions = np.random.randn(100, 3) * 2
   colors = np.random.randint(0, 255, size=(100, 3))
   
   # Much more efficient than 100 individual spheres
   server.scene.add_batched_axes(
       "many_frames",
       axes_lengths=np.full(100, 0.1),
       wxyzs=np.tile([1, 0, 0, 0], (100, 1)),  # No rotation
       positions=positions,
   )

**Level of Detail:**

.. code-block:: python

   # Show different detail levels based on distance or performance
   full_detail_points = np.random.randn(10000, 3)
   reduced_points = full_detail_points[::10]  # Every 10th point
   
   # Switch based on performance needs
   detail_level = "high"  # or "low"
   
   if detail_level == "high":
       server.scene.add_point_cloud("data", points=full_detail_points)
   else:
       server.scene.add_point_cloud("data", points=reduced_points)

Common Patterns
---------------

**Loading and Visualizing Data:**

.. code-block:: python

   def visualize_point_cloud(points, colors=None):
       if colors is None:
           # Color by height
           z_vals = points[:, 2]
           norm_z = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
           colors = np.column_stack([
               norm_z * 255,
               np.zeros_like(norm_z),
               (1 - norm_z) * 255
           ]).astype(np.uint8)
       
       server.scene.add_point_cloud("data", points=points, colors=colors)

**Interactive Object Selection:**

.. code-block:: python

   selected_object = None
   
   @server.scene.on_click
   def handle_selection(event):
       global selected_object
       
       # Deselect previous
       if selected_object:
           selected_object.color = (150, 150, 150)  # Gray
       
       # Select new
       selected_object = event.object
       if selected_object:
           selected_object.color = (255, 255, 0)  # Yellow highlight

**Camera Synchronization:**

.. code-block:: python

   # Show camera frustums that match the view
   camera_positions = [(2, 0, 1), (-2, 0, 1), (0, 2, 1)]
   
   for i, pos in enumerate(camera_positions):
       server.scene.add_camera_frustum(
           f"cam_{i}",
           fov=75,  # Match viewer FOV
           aspect=16/9,
           scale=0.3,
           position=pos,
           color=(255, 100, 100)
       )

API Reference
-------------

.. autoclass:: viser.SceneApi
   :members:
   :undoc-members:
   :inherited-members:
