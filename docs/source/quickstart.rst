5-Minute Quickstart
===================

Get up and running with viser in 5 minutes! This guide covers the essential patterns you'll use in most projects.

Step 1: Installation (30 seconds)
----------------------------------

.. code-block:: bash

   pip install viser[examples]

Step 2: Hello World (1 minute)
-------------------------------

Create ``demo.py``:

.. code-block:: python

   import viser

   server = viser.ViserServer()
   
   # Add a red sphere
   server.scene.add_icosphere(
       name="sphere",
       radius=0.5,
       color=(255, 0, 0),
       position=(0, 0, 0)
   )
   
   print("Visit http://localhost:8080")
   while True: pass

Run it and open the URL in your browser!

Step 3: Add Some GUI Controls (2 minutes) 
------------------------------------------

Let's make it interactive:

.. code-block:: python

   import viser
   import time

   server = viser.ViserServer()
   
   # Add sphere  
   sphere = server.scene.add_icosphere("sphere", radius=0.5, color=(255, 0, 0))
   
   # Add GUI controls
   with server.gui.add_folder("Sphere Controls"):
       radius_slider = server.gui.add_slider("Radius", min=0.1, max=2.0, step=0.1, initial_value=0.5)
       color_picker = server.gui.add_rgb("Color", initial_value=(255, 0, 0))
       visible_checkbox = server.gui.add_checkbox("Visible", initial_value=True)
   
   # Handle updates
   @radius_slider.on_update  
   def update_radius() -> None:
       sphere.radius = radius_slider.value
   
   @color_picker.on_update
   def update_color() -> None:
       sphere.color = color_picker.value
       
   @visible_checkbox.on_update
   def update_visibility() -> None:
       sphere.visible = visible_checkbox.value
   
   print("Visit http://localhost:8080")
   while True: 
       time.sleep(0.1)

Now you have an interactive sphere with live controls!

Step 4: Add More Objects (1 minute)
------------------------------------

Let's build a small scene:

.. code-block:: python

   import viser
   import numpy as np
   import time

   server = viser.ViserServer()
   
   # Add coordinate frame for reference
   server.scene.add_frame("world", axes_length=1.0, axes_radius=0.02)
   
   # Add various objects
   server.scene.add_icosphere("sphere", radius=0.3, color=(255, 0, 0), position=(1, 0, 0))
   server.scene.add_box("box", dimensions=(0.5, 0.5, 0.5), color=(0, 255, 0), position=(-1, 0, 0))
   server.scene.add_cylinder("cylinder", height=1.0, radius=0.2, color=(0, 0, 255), position=(0, 1, 0))
   
   # Add a point cloud
   points = np.random.randn(1000, 3) * 0.5
   colors = np.random.randint(0, 255, size=(1000, 3))
   server.scene.add_point_cloud("random_points", points=points, colors=colors, point_size=0.02)
   
   # Add some lines
   line_points = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0], [-1, 0, 1]])
   server.scene.add_line_segments("path", points=line_points[:-1], points_to=line_points[1:], color=(255, 255, 0))
   
   print("Visit http://localhost:8080 to see your 3D scene!")
   while True:
       time.sleep(0.1)

Step 5: Handle User Clicks (30 seconds)
----------------------------------------

Make objects respond to clicks:

.. code-block:: python

   @server.scene.on_click
   def handle_click(click_event: viser.ScenePointerEvent) -> None:
       print(f"Clicked on: {click_event.object_name}")
       
       # Change color when clicked
       if hasattr(click_event.object, 'color'):
           import random
           click_event.object.color = (
               random.randint(0, 255),
               random.randint(0, 255), 
               random.randint(0, 255)
           )

🎉 **Congratulations!** 
------------------------

In 5 minutes, you've learned:

- ✅ Creating a viser server
- ✅ Adding 3D objects (spheres, boxes, point clouds, lines)  
- ✅ Building interactive GUI controls
- ✅ Handling user input and events
- ✅ Creating dynamic, responsive visualizations

What's Next?
------------

Now that you have the basics, explore more advanced features:

**📖 Learn by Example**
   - :doc:`examples/01_scene_index` - Master 3D visualization fundamentals
   - :doc:`examples/02_gui_index` - Build sophisticated user interfaces  
   - :doc:`examples/03_interaction_index` - Create interactive applications
   - :doc:`examples/04_demos_index` - See real-world applications

**🛠️ Build Something**
   - Visualize your own data (robotics sensors, ML models, scientific data)
   - Create interactive tools for your team
   - Build data exploration dashboards
   - Prototype 3D applications

**📚 Go Deeper**  
   - :doc:`user_guides/index` - Task-oriented tutorials
   - :doc:`server` - Complete API reference
   - :doc:`best_practices` - Tips for building robust applications

Ready to dive deeper? Start with :doc:`examples/01_scene_index` to master the fundamentals!