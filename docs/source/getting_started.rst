Getting Started
===============

Welcome to viser! This guide will get you up and running with interactive 3D visualization in just a few minutes.

What is viser?
--------------

**viser** is a Python library for building interactive 3D visualizations with web-based clients. Perfect for:

- **Robotics**: Visualize robot models, sensor data, and trajectories
- **Machine Learning**: Interactive plots, 3D data exploration, model debugging  
- **Computer Vision**: Point clouds, camera poses, 3D reconstructions
- **Scientific Computing**: Any data that benefits from 3D visualization

Key features:

- 🎯 **Simple API** - Add 3D objects with just a few lines of code
- 🎛️ **Interactive GUI** - Build control panels with sliders, buttons, and inputs
- 🖱️ **User Interaction** - Handle clicks, selection, and scene manipulation
- 🌐 **Web-based** - Works over SSH, no desktop environment needed
- ⚡ **Real-time** - Live updates and streaming data visualization

Installation
------------

Install viser with pip:

.. code-block:: bash

   pip install viser

For running examples, install with example dependencies:

.. code-block:: bash

   pip install viser[examples]

That's it! No additional setup required.

Your First Visualization
-------------------------

Let's create your first 3D scene. Create a file called ``hello_viser.py``:

.. code-block:: python

   import viser

   server = viser.ViserServer()
   server.scene.add_icosphere(
       name="hello_sphere",
       radius=0.5,
       color=(255, 0, 0),  # Red
       position=(0.0, 0.0, 0.0),
   )

   print("Open your browser to http://localhost:8080")
   print("Press Ctrl+C to exit")

   while True:
       pass

Run it:

.. code-block:: bash

   python hello_viser.py

Open your browser to ``http://localhost:8080`` and you'll see a red sphere! 🎉

.. note::
   The visualization runs in your browser, so it works great over SSH or on remote servers.

What's Next?
------------

Now that you have viser running, here's your learning path:

1. **🎯 Scene Basics** (:doc:`examples/01_scene_index`) 
   
   Learn the fundamentals: coordinate systems, meshes, cameras, and lighting.
   
   Start with: :doc:`examples/00_coordinate_frames`

2. **🎛️ Interactive GUI** (:doc:`examples/02_gui_index`)
   
   Build control panels with sliders, buttons, and custom layouts.
   
   Start with: :doc:`examples/02_gui_00_basic_controls`

3. **🖱️ User Interaction** (:doc:`examples/03_interaction_index`)
   
   Handle mouse clicks and build interactive applications.
   
   Start with: :doc:`examples/12_click_meshes`

4. **🚀 Complete Applications** (:doc:`examples/04_demos_index`)
   
   See real-world examples with external tools and datasets.
   
   Start with: :doc:`examples/07_record3d_visualizer`

Key Concepts
------------

Understanding these concepts will help you get the most out of viser:

**Client-Server Architecture**
   Viser runs a Python server that communicates with a web client. This means your visualization logic stays in Python while the rendering happens in the browser.

**Scene Graph**
   Objects in your 3D scene are organized in a tree structure. You can group objects, apply transformations, and control visibility hierarchically.

**Coordinate Systems**
   Viser uses right-handed coordinates with Y-up by default. Understanding :doc:`conventions` will help you position objects correctly.

**Real-time Updates**
   Your Python code can modify the scene in real-time. Add objects, change properties, or respond to user input while the visualization is running.

Common Patterns
---------------

Here are some patterns you'll use frequently:

**Adding 3D Objects**

.. code-block:: python

   import numpy as np
   import viser

   server = viser.ViserServer()

   # Basic shapes
   server.scene.add_icosphere("sphere", radius=1.0, color=(255, 0, 0))
   server.scene.add_box("box", dimensions=(1, 2, 3), color=(0, 255, 0))
   server.scene.add_cylinder("cylinder", height=2.0, radius=0.5)

   # Coordinate frames
   server.scene.add_frame("frame", axes_length=1.0, axes_radius=0.02)

   # Point clouds and meshes
   points = np.random.randn(1000, 3)
   colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
   vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
   faces = np.array([[0, 1, 2]], dtype=np.uint32)
   
   server.scene.add_point_cloud("points", points=points, colors=colors)
   server.scene.add_mesh_simple("mesh", vertices=vertices, faces=faces)

**Building GUI Controls**

.. code-block:: python

   import viser

   server = viser.ViserServer()

   # Organize with folders
   with server.gui.add_folder("Controls"):
       radius_slider = server.gui.add_slider(
           "Radius", min=0.1, max=2.0, step=0.1, initial_value=1.0
       )
       color_picker = server.gui.add_rgb("Color", initial_value=(255, 0, 0))
       show_checkbox = server.gui.add_checkbox("Show Object", initial_value=True)

**Handling User Input**

.. code-block:: python

   import viser

   server = viser.ViserServer()
   sphere_handle = server.scene.add_icosphere("sphere", radius=1.0)
   radius_slider = server.gui.add_slider("Radius", min=0.1, max=2.0)

   @radius_slider.on_update
   def update_radius() -> None:
       sphere_handle.radius = radius_slider.value

   @server.scene.on_click
   def handle_click(click_event: viser.ScenePointerEvent) -> None:
       print(f"Clicked: {click_event.object_name}")

Need Help?
----------

- **📖 Examples**: Browse :doc:`examples/index` for code samples
- **📚 API Reference**: See :doc:`server` for complete API documentation  
- **🔧 Troubleshooting**: Check :doc:`troubleshooting` for common issues
- **💬 Community**: Join discussions on GitHub Issues

Ready to build something amazing? Start with the :doc:`examples/01_scene_index`!