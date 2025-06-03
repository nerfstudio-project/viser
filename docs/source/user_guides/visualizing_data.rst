Visualizing Data
================

Learn how to visualize different types of data with viser, from simple 3D objects to complex scientific datasets.

Point Clouds
------------

Point clouds are one of the most common data types in 3D visualization.

**Basic Point Cloud**

.. code-block:: python

   import numpy as np
   import viser
   
   server = viser.ViserServer()
   
   # Generate random points
   points = np.random.randn(1000, 3)
   colors = np.random.randint(0, 255, size=(1000, 3))
   
   server.scene.add_point_cloud(
       "my_points",
       points=points,
       colors=colors,
       point_size=0.02
   )

**From Real Data (e.g., LiDAR)**

.. code-block:: python

   # Load your point cloud data
   points = load_lidar_data("scan.pcd")  # Your loading function
   
   # Color by height (Z coordinate)
   z_min, z_max = points[:, 2].min(), points[:, 2].max()
   normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
   
   # Create colormap (blue to red)
   colors = np.zeros((len(points), 3), dtype=np.uint8)
   colors[:, 0] = (normalized_z * 255).astype(np.uint8)      # Red channel
   colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel
   
   server.scene.add_point_cloud("lidar_scan", points=points, colors=colors)

**Interactive Point Cloud Viewer**

.. code-block:: python

   import viser
   import numpy as np
   
   server = viser.ViserServer()
   
   # Load data
   points = np.random.randn(5000, 3)
   
   # Add point cloud
   pc_handle = server.scene.add_point_cloud("points", points=points, point_size=0.01)
   
   # Add controls
   with server.gui.add_folder("Point Cloud"):
       size_slider = server.gui.add_slider("Size", min=0.001, max=0.1, step=0.001, initial_value=0.01)
       color_mode = server.gui.add_dropdown("Color Mode", options=["Height", "Random", "Solid"])
       point_count = server.gui.add_slider("Point Count", min=100, max=len(points), step=100, initial_value=len(points))
   
   @size_slider.on_update
   def update_size() -> None:
       pc_handle.point_size = size_slider.value
   
   @color_mode.on_update
   def update_colors() -> None:
       n_points = int(point_count.value)
       if color_mode.value == "Height":
           # Color by Z coordinate
           z_vals = points[:n_points, 2]
           norm_z = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
           colors = np.column_stack([norm_z * 255, np.zeros(n_points), (1-norm_z) * 255]).astype(np.uint8)
       elif color_mode.value == "Random":
           colors = np.random.randint(0, 255, size=(n_points, 3))
       else:  # Solid
           colors = np.full((n_points, 3), [100, 150, 255], dtype=np.uint8)
       
       pc_handle.colors = colors
   
   @point_count.on_update  
   def update_count() -> None:
       n_points = int(point_count.value)
       pc_handle.points = points[:n_points]
       update_colors()  # Recompute colors for new point count

3D Meshes
---------

Visualize complex 3D geometry with meshes.

**Loading from Files**

.. code-block:: python

   import trimesh
   import viser
   
   server = viser.ViserServer()
   
   # Load mesh (supports .obj, .ply, .stl, etc.)
   mesh = trimesh.load_mesh("model.obj")
   
   server.scene.add_mesh_simple(
       "loaded_mesh",
       vertices=mesh.vertices,
       faces=mesh.faces,
       color=(150, 150, 150)
   )

**Procedural Meshes**

.. code-block:: python

   import numpy as np
   import viser
   
   def create_sphere_mesh(radius: float = 1.0, resolution: int = 20):
       """Create a sphere mesh programmatically."""
       phi = np.linspace(0, np.pi, resolution)
       theta = np.linspace(0, 2*np.pi, resolution)
       
       vertices = []
       for p in phi:
           for t in theta:
               x = radius * np.sin(p) * np.cos(t)
               y = radius * np.sin(p) * np.sin(t) 
               z = radius * np.cos(p)
               vertices.append([x, y, z])
       
       vertices = np.array(vertices)
       
       # Generate faces (triangulation)
       faces = []
       for i in range(resolution - 1):
           for j in range(resolution - 1):
               # Two triangles per quad
               v1 = i * resolution + j
               v2 = v1 + 1
               v3 = v1 + resolution
               v4 = v3 + 1
               
               faces.append([v1, v2, v3])
               faces.append([v2, v4, v3])
       
       return vertices, np.array(faces)
   
   server = viser.ViserServer()
   vertices, faces = create_sphere_mesh()
   server.scene.add_mesh_simple("sphere", vertices=vertices, faces=faces)

**Mesh with Materials**

.. code-block:: python

   # Advanced mesh with vertex colors
   mesh = trimesh.load_mesh("textured_model.obj")
   
   # Apply vertex coloring based on height
   z_coords = mesh.vertices[:, 2]
   normalized_z = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
   
   vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
   vertex_colors[:, 0] = (normalized_z * 255).astype(np.uint8)      # Red
   vertex_colors[:, 1] = ((1 - normalized_z) * 255).astype(np.uint8)  # Green
   
   server.scene.add_mesh_simple(
       "colored_mesh",
       vertices=mesh.vertices,
       faces=mesh.faces,
       vertex_colors=vertex_colors
   )

Time Series & Trajectories
--------------------------

Visualize data that changes over time.

**Robot Trajectory**

.. code-block:: python

   import numpy as np
   import viser
   import viser.transforms as tf
   import time
   
   server = viser.ViserServer()
   
   # Generate trajectory data
   t_vals = np.linspace(0, 4*np.pi, 100)
   positions = np.column_stack([
       np.cos(t_vals),
       np.sin(t_vals),
       0.1 * t_vals
   ])
   
   # Show full trajectory as line
   server.scene.add_line_segments(
       "trajectory",
       points=positions[:-1],
       points_to=positions[1:],
       color=(255, 255, 0),
       line_width=3.0
   )
   
   # Add robot at each pose
   robot_poses = []
   for i, pos in enumerate(positions[::5]):  # Every 5th point
       # Compute orientation (tangent to trajectory)
       if i < len(positions[::5]) - 1:
           direction = positions[::5][i+1] - pos
           direction = direction / np.linalg.norm(direction)
           
           # Create rotation matrix from direction
           up = np.array([0, 0, 1])
           right = np.cross(direction, up)
           up = np.cross(right, direction)
           
           rotation_matrix = np.column_stack([right, up, direction])
           rotation = tf.SO3.from_matrix(rotation_matrix)
       else:
           rotation = tf.SO3.identity()
       
       # Add robot frame
       server.scene.add_frame(
           f"robot_pose_{i}",
           axes_length=0.1,
           axes_radius=0.01,
           wxyz=rotation.wxyz,
           position=pos
       )

**Animated Visualization**

.. code-block:: python

   server = viser.ViserServer()
   
   # Create moving robot
   robot = server.scene.add_box(
       "robot",
       dimensions=(0.1, 0.1, 0.05),
       color=(0, 255, 0)
   )
   
   # Trail of previous positions
   trail_points = []
   trail_handle = None
   
   # Animation loop
   for i, pos in enumerate(positions):
       robot.position = pos
       
       # Update trail
       trail_points.append(pos)
       if len(trail_points) > 20:  # Keep last 20 points
           trail_points.pop(0)
       
       if len(trail_points) > 1:
           if trail_handle:
               trail_handle.remove()
           
           trail_handle = server.scene.add_line_segments(
               "trail",
               points=np.array(trail_points[:-1]),
               points_to=np.array(trail_points[1:]),
               color=(0, 255, 255),
               line_width=2.0
           )
       
       time.sleep(0.1)

Sensor Data
-----------

Visualize data from various sensors.

**Camera Poses & Images**

.. code-block:: python

   import viser
   import viser.transforms as tf
   import imageio.v3 as iio
   
   server = viser.ViserServer()
   
   # Camera parameters
   camera_positions = [
       [0, 0, 2],
       [1, 0, 2], 
       [0, 1, 2],
       [-1, 0, 2]
   ]
   
   # Add camera frustums
   for i, pos in enumerate(camera_positions):
       # Camera frustum
       server.scene.add_camera_frustum(
           f"camera_{i}",
           fov=60,
           aspect=1.33,
           scale=0.3,
           color=(255, 100, 100),
           position=pos,
           wxyz=tf.SO3.from_z_radians(np.pi).wxyz  # Point down
       )
       
       # Load and display image
       if i == 0:  # Just show image for first camera
           image = iio.imread("camera_image.jpg")
           server.scene.add_image(
               f"image_{i}",
               image=image,
               render_width=0.5,
               render_height=0.5 * image.shape[0] / image.shape[1],
               position=np.array(pos) + [0, 0, -0.5]
           )

**Multi-Modal Sensor Fusion**

.. code-block:: python

   import viser
   import numpy as np
   
   server = viser.ViserServer()
   
   # LiDAR data
   lidar_points = np.random.randn(1000, 3) * 5
   server.scene.add_point_cloud(
       "lidar",
       points=lidar_points,
       colors=np.full((1000, 3), [0, 255, 0]),  # Green
       point_size=0.02
   )
   
   # Radar detections (as spheres)
   radar_detections = np.random.randn(20, 3) * 3
   for i, detection in enumerate(radar_detections):
       server.scene.add_icosphere(
           f"radar_{i}",
           radius=0.1,
           color=(255, 0, 0),  # Red
           position=detection
       )
   
   # GPS trajectory
   gps_waypoints = np.random.randn(50, 3) * 2
   gps_waypoints[:, 2] = 0  # Keep on ground
   
   server.scene.add_line_segments(
       "gps_path",
       points=gps_waypoints[:-1],
       points_to=gps_waypoints[1:],
       color=(0, 0, 255),  # Blue
       line_width=3.0
   )

Scientific Data
---------------

Visualize complex scientific datasets.

**Volumetric Data**

.. code-block:: python

   import numpy as np
   import viser
   
   server = viser.ViserServer()
   
   # Create sample volume data (e.g., MRI scan, fluid simulation)
   resolution = 32
   x, y, z = np.meshgrid(
       np.linspace(-2, 2, resolution),
       np.linspace(-2, 2, resolution), 
       np.linspace(-2, 2, resolution)
   )
   
   # Example: distance field of spheres
   volume_data = np.sqrt(x**2 + y**2 + z**2)
   
   # Extract isosurface (marching cubes)
   from skimage import measure
   vertices, faces, _, _ = measure.marching_cubes(volume_data, level=1.0)
   
   # Scale vertices to world coordinates
   vertices = vertices / resolution * 4 - 2
   
   server.scene.add_mesh_simple(
       "isosurface",
       vertices=vertices,
       faces=faces,
       color=(100, 200, 255)
   )

**Multi-Dimensional Data**

.. code-block:: python

   # Visualize high-dimensional data with dimensionality reduction
   from sklearn.decomposition import PCA
   import numpy as np
   
   # High-dimensional data (e.g., features from ML model)
   high_dim_data = np.random.randn(1000, 50)  # 1000 samples, 50 features
   labels = np.random.randint(0, 3, 1000)      # 3 classes
   
   # Reduce to 3D
   pca = PCA(n_components=3)
   points_3d = pca.fit_transform(high_dim_data)
   
   # Color by class
   colors = np.array([
       [255, 0, 0],    # Red for class 0
       [0, 255, 0],    # Green for class 1  
       [0, 0, 255]     # Blue for class 2
   ])[labels]
   
   server.scene.add_point_cloud(
       "high_dim_projection",
       points=points_3d,
       colors=colors,
       point_size=0.05
   )
   
   # Add explained variance information
   variance_explained = pca.explained_variance_ratio_
   with server.gui.add_folder("PCA Info"):
       server.gui.add_text("PC1 Variance", f"{variance_explained[0]:.3f}")
       server.gui.add_text("PC2 Variance", f"{variance_explained[1]:.3f}")
       server.gui.add_text("PC3 Variance", f"{variance_explained[2]:.3f}")
       server.gui.add_text("Total Explained", f"{variance_explained.sum():.3f}")

Performance Tips
----------------

**For Large Datasets:**

1. **Decimation** - Reduce point count intelligently

.. code-block:: python

   # Voxel-based decimation
   def decimate_points(points, voxel_size: float = 0.1):
       # Simple voxel grid decimation
       voxel_coords = np.floor(points / voxel_size).astype(int)
       _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
       return points[unique_indices]
   
   # Apply decimation for display
   decimated_points = decimate_points(large_point_cloud, voxel_size=0.05)
   server.scene.add_point_cloud("decimated", points=decimated_points)

2. **Level of Detail** - Show different detail levels based on distance

.. code-block:: python

   # Multiple LOD versions
   full_detail = points
   medium_detail = points[::2]  # Every other point
   low_detail = points[::10]    # Every 10th point
   
   # Switch based on camera distance or user preference
   with server.gui.add_folder("Level of Detail"):
       lod_level = server.gui.add_dropdown("Detail", options=["High", "Medium", "Low"])
   
   @lod_level.on_update
   def update_lod() -> None:
       pc_handle.remove()
       if lod_level.value == "High":
           pc_handle = server.scene.add_point_cloud("points", points=full_detail)
       elif lod_level.value == "Medium":
           pc_handle = server.scene.add_point_cloud("points", points=medium_detail)
       else:
           pc_handle = server.scene.add_point_cloud("points", points=low_detail)

3. **Streaming Updates** - Update data incrementally

.. code-block:: python

   # For real-time data streams
   import queue
   import threading
   
   data_queue = queue.Queue()
   
   def data_producer() -> None:
       """Simulate real-time data source."""
       while True:
           new_points = generate_new_data()  # Your data source
           data_queue.put(new_points)
           time.sleep(0.1)
   
   # Start data producer in background
   threading.Thread(target=data_producer, daemon=True).start()
   
   # Main visualization loop
   accumulated_points = []
   while True:
       try:
           new_points = data_queue.get_nowait()
           accumulated_points.extend(new_points)
           
           # Keep only recent points
           if len(accumulated_points) > 10000:
               accumulated_points = accumulated_points[-10000:]
           
           # Update visualization
           pc_handle.points = np.array(accumulated_points)
           
       except queue.Empty:
           pass
       
       time.sleep(0.05)

Next Steps
----------

- **Try it yourself**: Start with :doc:`../examples/01_scene_index` 
- **Build interfaces**: Learn :doc:`building_gui`
- **Optimize performance**: Read :doc:`performance`
- **Domain-specific guides**: Check :doc:`domain_specific`