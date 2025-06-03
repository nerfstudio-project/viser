Troubleshooting
===============

Common issues and solutions when working with viser.

Installation Issues
-------------------

**"No module named 'viser'" after installation**

.. code-block:: bash

   # Make sure you're using the right Python environment
   python -m pip install viser
   
   # Or if using conda:
   conda install -c conda-forge viser

**Import errors with example dependencies**

.. code-block:: bash

   # Install with example dependencies
   pip install "viser[examples]"
   
   # Or install missing packages individually:
   pip install trimesh imageio plotly

**Permission errors during installation**

.. code-block:: bash

   # Use --user flag to install in user directory
   pip install --user viser
   
   # Or use a virtual environment:
   python -m venv viser_env
   source viser_env/bin/activate  # On Windows: viser_env\Scripts\activate
   pip install viser

Connection Issues
-----------------

**"Connection refused" or can't access localhost:8080**

1. **Check if server is running:**

   .. code-block:: python

      import viser
      
      server = viser.ViserServer()
      print(f"Server running on http://localhost:{server.port}")
      
      # Keep server alive
      while True:
          pass

2. **Try a different port:**

   .. code-block:: python

      server = viser.ViserServer(port=8081)

3. **Check firewall settings:**
   
   - Ensure port 8080 (or your chosen port) is not blocked
   - Some corporate networks block certain ports

**Remote access issues (SSH/server deployment)**

.. code-block:: python

   # Allow connections from any IP
   server = viser.ViserServer(host="0.0.0.0", port=8080)

.. code-block:: bash

   # SSH port forwarding
   ssh -L 8080:localhost:8080 user@remote-server

**Browser compatibility issues**

- **Modern browsers required**: Chrome 91+, Firefox 90+, Safari 14+
- **WebSocket support**: Ensure WebSockets are enabled
- **HTTPS issues**: Some features require HTTPS in production

Performance Issues
------------------

**Slow rendering with large datasets**

1. **Reduce point count:**

   .. code-block:: python

      # Decimate large point clouds
      points = large_point_cloud[::10]  # Every 10th point
      server.scene.add_point_cloud("decimated", points=points)

2. **Use level of detail:**

   .. code-block:: python

      # Show different detail based on performance
      if len(points) > 100000:
          points = points[::5]  # Reduce detail for large datasets
      
      server.scene.add_point_cloud("adaptive", points=points)

3. **Batch similar objects:**

   .. code-block:: python

      # Instead of many individual spheres
      positions = np.random.randn(1000, 3)
      colors = np.random.randint(0, 255, (1000, 3))
      
      # Use batched rendering
      server.scene.add_batched_axes(
          "many_objects",
          axes_lengths=np.full(1000, 0.1),
          positions=positions
      )

**High memory usage**

.. code-block:: python

   # Remove unused objects
   old_handle.remove()
   
   # Clear entire scene
   server.scene.reset()
   
   # Use appropriate data types
   points = points.astype(np.float32)  # Instead of float64
   colors = colors.astype(np.uint8)    # Instead of int32

**Slow GUI updates**

.. code-block:: python

   # Debounce rapid updates
   import time
   
   last_update = 0
   
   @slider.on_update
   def debounced_update():
       global last_update
       current_time = time.time()
       
       if current_time - last_update > 0.1:  # 100ms debounce
           expensive_operation()
           last_update = current_time

Visualization Issues
--------------------

**Objects not appearing**

1. **Check object positioning:**

   .. code-block:: python

      # Add coordinate frame for reference
      server.scene.add_frame("world", axes_length=1.0)
      
      # Check if objects are at origin
      print(f"Object position: {object_handle.position}")

2. **Verify scale:**

   .. code-block:: python

      # Objects might be too small/large
      server.scene.add_icosphere("test", radius=1.0, color=(255, 0, 0))  # Visible size

3. **Check visibility:**

   .. code-block:: python

      # Ensure object is visible
      object_handle.visible = True

4. **Inspect scene hierarchy:**

   .. code-block:: python

      # List all scene objects
      for name in server.scene._handles:
          print(f"Object: {name}")

**Colors not displaying correctly**

.. code-block:: python

   # Ensure correct color format
   
   # RGB values 0-255 (integers)
   color = (255, 0, 0)  # ✓ Correct
   
   # NOT 0-1 floats (common mistake)
   color = (1.0, 0.0, 0.0)  # ✗ Will appear very dark
   
   # For arrays, use uint8
   colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)

**Coordinate system confusion**

.. code-block:: python

   # Viser uses Y-up, right-handed coordinates
   
   #     Y (up)
   #     │
   #     │
   #     └─────── X (right)
   #    ╱
   #   ╱  
   #  Z (forward, toward viewer)
   
   # If coming from Z-up systems, you may need to rotate:
   import viser.transforms as tf
   
   # Rotate from Z-up to Y-up
   z_to_y_rotation = tf.SO3.from_x_radians(-np.pi/2)
   server.scene.add_mesh_simple(
       "rotated_mesh",
       vertices=vertices,
       faces=faces,
       wxyz=z_to_y_rotation.wxyz
   )

**Camera/view issues**

.. code-block:: python

   # Reset camera to default view
   server.camera.position = (3, 3, 3)
   server.camera.look_at = (0, 0, 0)
   
   # Fit camera to scene
   server.camera.up_direction = (0, 1, 0)

GUI Issues
----------

**Controls not responding**

1. **Check event handlers:**

   .. code-block:: python

      @button.on_click
      def handle_click():
          print("Button clicked!")  # Debug output
      
      @slider.on_update
      def handle_update():
          print(f"Slider value: {slider.value}")  # Debug output

2. **Verify control creation:**

   .. code-block:: python

      # Make sure controls are created properly
      slider = server.gui.add_slider("Test", min=0, max=100, initial_value=50)
      print(f"Slider created: {slider}")

**Modal/folder issues**

.. code-block:: python

   # Ensure proper context management
   with server.gui.add_modal("Settings") as modal:
       # Add controls here
       button = server.gui.add_button("Close")
       
       @button.on_click
       def close_modal():
           modal.close()  # Proper cleanup

**State synchronization issues**

.. code-block:: python

   # Ensure UI and scene stay in sync
   def update_everything():
       # Update scene object
       sphere.radius = radius_slider.value
       sphere.color = color_picker.value
       
       # Update related UI elements
       status_text.value = f"Radius: {radius_slider.value:.2f}"
   
   radius_slider.on_update(update_everything)
   color_picker.on_update(update_everything)

Data Loading Issues
-------------------

**File not found errors**

.. code-block:: python

   import os
   from pathlib import Path
   
   # Use absolute paths
   file_path = Path(__file__).parent / "assets" / "model.obj"
   
   if not file_path.exists():
       print(f"File not found: {file_path}")
       print(f"Current directory: {os.getcwd()}")
       print(f"Files in assets: {list(Path('assets').glob('*'))}")
   else:
       mesh = trimesh.load_mesh(str(file_path))

**Mesh loading errors**

.. code-block:: python

   try:
       mesh = trimesh.load_mesh("model.obj")
       
       # Validate mesh
       if not mesh.is_valid:
           mesh.fix_normals()
       
       print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
       
   except Exception as e:
       print(f"Error loading mesh: {e}")
       # Use fallback geometry
       vertices, faces = trimesh.creation.icosphere().vertices, trimesh.creation.icosphere().faces

**Point cloud format issues**

.. code-block:: python

   # Ensure correct array shapes and types
   points = np.array(points, dtype=np.float32)
   
   if points.ndim != 2 or points.shape[1] != 3:
       raise ValueError(f"Points must be (N, 3) array, got {points.shape}")
   
   if colors is not None:
       colors = np.array(colors, dtype=np.uint8)
       if colors.shape != (len(points), 3):
           raise ValueError(f"Colors must be (N, 3) array, got {colors.shape}")

Development Issues
------------------

**Hot reloading during development**

.. code-block:: python

   # For development, automatically restart on file changes
   import sys
   import importlib
   
   def reload_modules():
       for module_name in list(sys.modules.keys()):
           if module_name.startswith('your_project'):
               importlib.reload(sys.modules[module_name])

**Debugging techniques**

.. code-block:: python

   # Add debug information to GUI
   with server.gui.add_folder("Debug"):
       fps_display = server.gui.add_text("FPS", "0", disabled=True)
       object_count = server.gui.add_text("Objects", "0", disabled=True)
   
   import time
   frame_times = []
   
   while True:
       start_time = time.time()
       
       # Your update code here
       
       # Update debug info
       frame_time = time.time() - start_time
       frame_times.append(frame_time)
       if len(frame_times) > 30:
           frame_times.pop(0)
       
       fps = 1.0 / np.mean(frame_times) if frame_times else 0
       fps_display.value = f"{fps:.1f}"
       object_count.value = str(len(server.scene._handles))
       
       time.sleep(0.033)  # ~30 FPS

**Memory debugging**

.. code-block:: python

   import psutil
   import os
   
   def get_memory_usage():
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / 1024 / 1024  # MB
   
   print(f"Memory usage: {get_memory_usage():.1f} MB")

Common Error Messages
---------------------

**"WebSocket connection failed"**
   - Server not running or wrong port
   - Firewall blocking connection  
   - Browser WebSocket support disabled

**"Object with name 'xyz' already exists"**
   - Remove existing object first: ``server.scene["xyz"].remove()``
   - Use unique names for each object

**"Invalid color format"**
   - Use RGB integers 0-255: ``(255, 0, 0)``
   - For arrays: ``np.uint8`` type

**"Array shape mismatch"**
   - Points: must be ``(N, 3)`` shape
   - Colors: must be ``(N, 3)`` shape matching points
   - Faces: must be ``(M, 3)`` for triangular meshes

**"AttributeError: 'NoneType' object has no attribute..."**
   - Object handle was removed or not created properly
   - Check return value of ``add_*`` methods

Getting Help
------------

If you're still having issues:

1. **Check examples**: Look for similar usage in :doc:`examples/index`
2. **Read the API docs**: See :doc:`server` for complete documentation
3. **Search GitHub issues**: `viser issues <https://github.com/nerfstudio-project/viser/issues>`_
4. **Create a minimal example**: Reproduce the issue with minimal code
5. **File a bug report**: Include Python version, browser, and minimal reproduction code

**When reporting bugs, include:**

.. code-block:: python

   # System information
   import sys
   import viser
   print(f"Python: {sys.version}")
   print(f"Viser: {viser.__version__}")
   print(f"Platform: {sys.platform}")

   # Minimal reproduction code
   server = viser.ViserServer()
   # ... minimal code that shows the issue

Performance Checklist
----------------------

Before reporting performance issues, try:

- ✅ Reduce dataset size (decimate point clouds, simplify meshes)
- ✅ Use appropriate data types (``float32``, ``uint8``)
- ✅ Remove unused objects from scene
- ✅ Debounce rapid GUI updates
- ✅ Use batching for many similar objects
- ✅ Check browser performance (try different browser)
- ✅ Monitor memory usage (browser dev tools)
- ✅ Test on different hardware if available

Most performance issues can be solved by reducing the amount of data being visualized or using more efficient rendering techniques.