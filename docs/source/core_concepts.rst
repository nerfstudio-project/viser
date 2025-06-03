Core Concepts
=============

Understanding these fundamental concepts will help you build better visualizations with viser.

Architecture Overview
---------------------

Viser uses a **client-server architecture** where your Python code runs the server and visualizations render in a web browser client.

.. code-block:: text

   ┌─────────────────┐    WebSocket     ┌──────────────────┐
   │  Python Server  │ ◄──────────────► │   Browser Client │
   │                 │                  │                  │
   │ • Your logic    │                  │ • 3D rendering   │
   │ • Data processing│                  │ • User interface │
   │ • Scene updates │                  │ • Event handling │
   └─────────────────┘                  └──────────────────┘

**Benefits:**
- 🌐 **Remote visualization** - Works over SSH, no desktop needed
- ⚡ **Real-time updates** - Python can modify the scene while rendering
- 🔧 **Easy deployment** - Just share a URL
- 💻 **Cross-platform** - Runs anywhere with a web browser

Scene Graph
-----------

Objects in viser are organized in a **hierarchical scene graph** - a tree structure where transformations propagate from parent to child nodes.

.. code-block:: python

   # Create a robot arm hierarchy
   base = server.scene.add_box("robot/base", dimensions=(0.2, 0.2, 0.1))
   
   # Shoulder joint (child of base)
   shoulder = server.scene.add_cylinder(
       "robot/base/shoulder", 
       height=0.3, 
       radius=0.05,
       position=(0, 0, 0.15)  # Relative to base
   )
   
   # Elbow joint (child of shoulder) 
   elbow = server.scene.add_cylinder(
       "robot/base/shoulder/elbow",
       height=0.2,
       radius=0.04, 
       position=(0, 0, 0.3)  # Relative to shoulder
   )

**Key Benefits:**
- 🔄 **Transform propagation** - Move the base, everything moves
- 📁 **Logical organization** - Group related objects
- 👁️ **Batch operations** - Show/hide entire groups
- 🎯 **Easy manipulation** - Transform gizmos work on groups

Coordinate Systems  
------------------

Viser uses **right-handed coordinates** with **Y-up** as the default:

.. code-block:: text

        Y (up)
        │
        │
        └─────── X (right)
       ╱
      ╱  
     Z (forward, toward viewer)

**Important Notes:**
- **Units**: Viser is unit-agnostic - use meters, millimeters, whatever fits your data
- **Rotations**: Use quaternions (w, x, y, z) or rotation matrices
- **Transforms**: Position + rotation + scale applied in that order

.. code-block:: python

   # Position in 3D space
   server.scene.add_icosphere(
       "sphere",
       radius=0.1,
       position=(1.0, 2.0, 3.0)  # (x, y, z)
   )
   
   # Rotation using quaternions (w, x, y, z)
   import viser.transforms as tf
   rotation = tf.SO3.from_x_radians(3.14159 / 4)  # 45 degrees around X
   
   server.scene.add_box(
       "rotated_box",
       dimensions=(1, 1, 1),
       wxyz=rotation.wxyz,  # Apply rotation
       position=(0, 0, 0)
   )

Real-time Updates
-----------------

One of viser's key strengths is **real-time scene modification**. Your Python code can continuously update the visualization.

.. code-block:: python

   import time
   import numpy as np
   
   server = viser.ViserServer()
   
   # Create an animated sphere
   sphere = server.scene.add_icosphere("moving_sphere", radius=0.1, color=(255, 0, 0))
   
   # Animation loop
   t = 0
   while True:
       # Update position in real-time
       x = np.cos(t)
       y = np.sin(t) 
       z = 0.5 * np.sin(2 * t)
       
       sphere.position = (x, y, z)
       sphere.color = (
           int(127 + 127 * np.cos(t)),
           int(127 + 127 * np.sin(t)), 
           int(127 + 127 * np.cos(2*t))
       )
       
       t += 0.1
       time.sleep(0.05)  # ~20 FPS

**Performance Tips:**
- 🎯 **Batch updates** - Group multiple changes together
- ⏱️ **Reasonable frame rates** - 10-30 FPS is usually sufficient
- 📊 **Limit data size** - Large point clouds/meshes impact performance
- 🔧 **Use handles** - Store object references for efficient updates

GUI Integration
---------------

Viser's GUI system lets you build control panels that integrate seamlessly with your 3D scene.

**Folder Organization**

.. code-block:: python

   with server.gui.add_folder("Visualization"):
       with server.gui.add_folder("Objects"):
           show_points = server.gui.add_checkbox("Show Points", initial_value=True)
           point_size = server.gui.add_slider("Point Size", min=0.01, max=0.1, step=0.01, initial_value=0.05)
       
       with server.gui.add_folder("Camera"):
           fov = server.gui.add_slider("Field of View", min=30, max=120, step=5, initial_value=75)
           camera_speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)

**Event Handling**

.. code-block:: python

   @point_size.on_update
   def update_point_size() -> None:
       point_cloud_handle.point_size = point_size.value
   
   @show_points.on_update  
   def toggle_points() -> None:
       point_cloud_handle.visible = show_points.value

**Advanced Patterns**

.. code-block:: python

   # Modal dialogs for complex operations
   with server.gui.add_modal("Settings") as modal:
       resolution = server.gui.add_dropdown("Resolution", options=["720p", "1080p", "4K"])
       quality = server.gui.add_slider("Quality", min=1, max=10, step=1, initial_value=5)
       
       confirm_button = server.gui.add_button("Apply Settings")
       
       @confirm_button.on_click
       def apply_settings() -> None:
           # Apply configuration
           modal.close()

Event System
------------

Viser provides a comprehensive event system for user interaction:

**Scene Events**

.. code-block:: python

   @server.scene.on_click  
   def handle_scene_click(event: viser.ScenePointerEvent) -> None:
       print(f"Clicked object: {event.object_name}")
       print(f"Click position: {event.click_pos}")
       print(f"Ray direction: {event.ray_direction}")

**GUI Events**

.. code-block:: python

   @button.on_click
   def handle_button_click() -> None:
       print("Button clicked!")
   
   @slider.on_update
   def handle_slider_change() -> None:
       print(f"New value: {slider.value}")

**Client Events**

.. code-block:: python

   @server.on_client_connect
   def welcome_user(client: viser.ClientHandle) -> None:
       print(f"New client connected: {client.client_id}")
   
   @server.on_client_disconnect  
   def goodbye_user(client: viser.ClientHandle) -> None:
       print(f"Client disconnected: {client.client_id}")

Data Flow Patterns
------------------

Understanding common data flow patterns will help you architect your applications:

**1. Reactive Updates**

.. code-block:: python

   # GUI controls drive scene updates
   @gui_control.on_update
   def update_scene() -> None:
       scene_object.property = gui_control.value

**2. Data Processing Pipeline**

.. code-block:: python

   def process_and_visualize(raw_data) -> None:
       # 1. Process data
       processed = preprocess(raw_data)
       
       # 2. Update visualization
       point_cloud.points = processed.points
       point_cloud.colors = processed.colors
       
       # 3. Update GUI to reflect changes
       gui_status.text = f"Processed {len(processed.points)} points"

**3. Multi-client Synchronization**

.. code-block:: python

   # Share state across multiple connected clients
   shared_state = {"current_frame": 0}
   
   @frame_slider.on_update
   def sync_frame() -> None:
       shared_state["current_frame"] = frame_slider.value
       update_all_clients(shared_state)

Best Practices
--------------

**🎯 Keep it responsive**
   - Update at reasonable frame rates (10-30 FPS)
   - Use smooth animations for property changes
   - Provide visual feedback for user actions

**📁 Organize logically**
   - Use meaningful object names (`robot/arm/joint1` not `object_42`)
   - Group related objects in the scene graph
   - Organize GUI controls in folders

**🔧 Handle errors gracefully**
   - Check for valid data before visualization
   - Provide helpful error messages in the GUI
   - Use try/except blocks around update code

**⚡ Optimize performance**
   - Reuse object handles instead of recreating
   - Batch multiple updates together
   - Consider data decimation for large datasets

Next Steps
----------

Now that you understand the core concepts:

1. **Practice with examples** - :doc:`examples/index`
2. **Build a simple project** - Start with your own data
3. **Explore advanced features** - :doc:`user_guides/index`
4. **Read the API reference** - :doc:`server`

Ready to start building? Check out :doc:`examples/01_scene_index` for hands-on practice!