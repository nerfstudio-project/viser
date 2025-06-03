GUI Controls
============

Learn to build interactive user interfaces that complement your 3D visualizations. Transform static scenes into dynamic, user-controlled applications.

Prerequisites
-------------

Before starting these examples, you should understand:

- ✅ **Scene Basics** - How to add 3D objects (see ``examples_dev/01_scene/``)
- ✅ **Coordinate Systems** - Positioning objects in 3D space

Learning Objectives
-------------------

These examples teach you:

- ✅ **Basic Controls** - Sliders, buttons, checkboxes, and dropdowns
- ✅ **Event Handling** - Responding to user input
- ✅ **Layout & Organization** - Folders, tabs, and logical grouping
- ✅ **Advanced UI** - Modals, notifications, and dynamic interfaces
- ✅ **Data Integration** - Connecting UI controls to visualizations
- ✅ **Custom Styling** - Theming and visual customization

Available Examples
------------------

Browse the ``examples_dev/02_gui/`` directory to find these examples:

.. raw:: html

   <style>
   .example-gallery {
       display: grid;
       grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
       gap: 20px;
       margin: 20px 0;
   }
   .example-card {
       border: 1px solid #e0e0e0;
       border-radius: 8px;
       overflow: hidden;
       transition: all 0.3s ease;
       background: white;
       text-decoration: none;
       display: block;
       color: inherit;
   }
   .example-card:hover {
       transform: translateY(-4px);
       box-shadow: 0 8px 16px rgba(0,0,0,0.1);
       border-color: #4A90E2;
   }
   .example-card img {
       width: 100%;
       height: 200px;
       object-fit: cover;
       background: #f5f5f5;
   }
   .example-card-content {
       padding: 15px;
   }
   .example-card h4 {
       margin: 0 0 8px 0;
       font-size: 14px;
       font-family: monospace;
       color: #333;
   }
   .example-card p {
       margin: 0;
       color: #666;
       font-size: 13px;
       line-height: 1.4;
   }
   </style>

   <div class="example-gallery">
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/00_basic_controls.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_00_basic_controls.png" alt="Basic Controls">
           <div class="example-card-content">
               <h4>00_basic_controls.py</h4>
               <p>🟢 Essential UI elements</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/01_callbacks.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_01_callbacks.png" alt="Callbacks">
           <div class="example-card-content">
               <h4>01_callbacks.py</h4>
               <p>🟢 Responding to user input</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/02_layouts.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_02_layouts.png" alt="Layouts">
           <div class="example-card-content">
               <h4>02_layouts.py</h4>
               <p>🟢 Organizing controls</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/03_markdown.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_03_markdown.png" alt="Markdown">
           <div class="example-card-content">
               <h4>03_markdown.py</h4>
               <p>🟡 Rich text formatting</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/04_modals.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_04_modals.png" alt="Modals">
           <div class="example-card-content">
               <h4>04_modals.py</h4>
               <p>🟡 Pop-up dialogs</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/05_theming.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_05_theming.png" alt="Theming">
           <div class="example-card-content">
               <h4>05_theming.py</h4>
               <p>🔴 Custom styling</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/06_gui_in_scene.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_06_gui_in_scene.png" alt="GUI in Scene">
           <div class="example-card-content">
               <h4>06_gui_in_scene.py</h4>
               <p>🔴 3D-embedded UI</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/07_notifications.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_07_notifications.png" alt="Notifications">
           <div class="example-card-content">
               <h4>07_notifications.py</h4>
               <p>🟡 User feedback</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/08_plotly_integration.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_08_plotly_integration.png" alt="Plotly">
           <div class="example-card-content">
               <h4>08_plotly_integration.py</h4>
               <p>🔴 Interactive plots</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/02_gui/09_plots_as_images.py" class="example-card" target="_blank">
           <img src="../../_static/examples/02_gui_09_plots_as_images.png" alt="Plot Images">
           <div class="example-card-content">
               <h4>09_plots_as_images.py</h4>
               <p>🔴 Static visualizations</p>
           </div>
       </a>
   </div>

Running the Examples
---------------------

.. code-block:: bash

   # Navigate to examples directory
   cd viser/examples_dev
   
   # Run any GUI example
   python 02_gui/00_basic_controls.py
   python 02_gui/02_layouts.py
   
   # Open browser to http://localhost:8080

Key Concepts
------------

**Folder Organization**
   Group related controls together using ``server.gui.add_folder()``. This keeps complex interfaces manageable.

**Event-Driven Updates**
   Use ``@control.on_update`` decorators to respond to user input and update your visualization in real-time.

**State Management**
   Maintain application state that synchronizes between GUI controls and 3D scene objects.

**Progressive Enhancement**
   Start with basic controls, then add advanced features like modals and custom themes.

Common Patterns
---------------

**Basic Control Setup:**

.. code-block:: python

   import viser
   
   server = viser.ViserServer()
   
   # Create controls
   with server.gui.add_folder("Settings"):
       size_slider = server.gui.add_slider("Size", min=0.1, max=2.0, step=0.1, initial_value=1.0)
       color_picker = server.gui.add_rgb("Color", initial_value=(255, 0, 0))
       visible_checkbox = server.gui.add_checkbox("Visible", initial_value=True)

**Event Handling:**

.. code-block:: python

   # Create a sphere to control
   sphere = server.scene.add_icosphere("sphere", radius=1.0, color=(255, 0, 0))
   
   # Connect controls to scene objects
   @size_slider.on_update
   def update_size():
       sphere.radius = size_slider.value
   
   @color_picker.on_update
   def update_color():
       sphere.color = color_picker.value
   
   @visible_checkbox.on_update
   def update_visibility():
       sphere.visible = visible_checkbox.value

**Dynamic Interfaces:**

.. code-block:: python

   # Controls that change based on user selection
   mode_dropdown = server.gui.add_dropdown("Mode", options=["Sphere", "Box", "Cylinder"])
   
   current_object = None
   current_controls = []
   
   @mode_dropdown.on_update
   def switch_mode():
       global current_object, current_controls
       
       # Remove old object and controls
       if current_object:
           current_object.remove()
       for control in current_controls:
           control.remove()
       current_controls.clear()
       
       # Add new object and controls based on selection
       if mode_dropdown.value == "Sphere":
           current_object = server.scene.add_icosphere("object", radius=0.5)
           current_controls.append(
               server.gui.add_slider("Radius", min=0.1, max=2.0, step=0.1, initial_value=0.5)
           )
       elif mode_dropdown.value == "Box":
           current_object = server.scene.add_box("object", dimensions=(1, 1, 1))
           current_controls.append(
               server.gui.add_vector3("Dimensions", initial_value=(1.0, 1.0, 1.0))
           )

UI Design Principles
--------------------

**1. Logical Grouping**
   - Group related controls in folders
   - Use clear, descriptive names
   - Organize by function, not by control type

**2. Immediate Feedback**
   - Update visualizations as users interact
   - Provide visual confirmation of actions
   - Show progress for long operations

**3. Progressive Disclosure**
   - Start with essential controls visible
   - Hide advanced options in collapsed folders
   - Use modals for complex configurations

**4. Consistent Layout**
   - Use similar patterns throughout your interface
   - Maintain consistent spacing and sizing
   - Follow platform conventions

Integration with 3D Scene
--------------------------

**Coordinated Updates:**

.. code-block:: python

   # Multiple controls affecting the same object
   def update_sphere():
       sphere.radius = size_slider.value
       sphere.color = color_picker.value
       sphere.position = position_vector.value
       sphere.visible = visible_checkbox.value
   
   # Connect all controls to the same update function
   size_slider.on_update(update_sphere)
   color_picker.on_update(update_sphere)
   position_vector.on_update(update_sphere)
   visible_checkbox.on_update(update_sphere)

**Scene-Driven UI:**

.. code-block:: python

   # Update UI based on scene interactions
   @server.scene.on_click
   def handle_object_click(event):
       # Update UI to show properties of clicked object
       if event.object_name == "sphere":
           size_slider.value = sphere.radius
           color_picker.value = sphere.color
           visible_checkbox.value = sphere.visible

Performance Tips
----------------

**Debounce Rapid Updates:**

.. code-block:: python

   import time
   
   last_update = 0
   
   @expensive_slider.on_update
   def debounced_update():
       global last_update
       current_time = time.time()
       if current_time - last_update > 0.1:  # 100ms debounce
           expensive_computation()
           last_update = current_time

**Batch Updates:**

.. code-block:: python

   # Update multiple properties together
   def update_all_properties():
       with server.atomic():  # Batch multiple updates
           sphere.radius = size_slider.value
           sphere.color = color_picker.value
           sphere.position = position_vector.value

Next Steps
----------

After mastering GUI controls:

1. **Add Interaction** → Browse ``examples_dev/03_interaction/`` to handle clicks and events
2. **Optimize Performance** → See :doc:`../../user_guides/performance` to handle large datasets
3. **See Complete Apps** → Explore ``examples_dev/04_demos/`` for real-world examples

**Ready to build interfaces?** Run ``python examples_dev/02_gui/00_basic_controls.py``!