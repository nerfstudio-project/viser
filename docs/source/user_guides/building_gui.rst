Building GUI
============

Learn how to create intuitive, responsive user interfaces that complement your 3D visualizations.

Basic Controls
--------------

Start with the fundamental GUI elements that every interface needs.

**Essential Controls**

.. code-block:: python

   import viser
   
   server = viser.ViserServer()
   
   with server.gui.add_folder("Basic Controls"):
       # Numeric inputs
       slider = server.gui.add_slider("Value", min=0, max=100, step=1, initial_value=50)
       number = server.gui.add_number("Precise", initial_value=3.14159, step=0.001)
       
       # Text inputs
       text_input = server.gui.add_text("Name", initial_value="Enter text here")
       
       # Selections
       checkbox = server.gui.add_checkbox("Enable Feature", initial_value=True)
       dropdown = server.gui.add_dropdown("Mode", options=["Option A", "Option B", "Option C"])
       
       # Colors
       color_picker = server.gui.add_rgb("Color", initial_value=(255, 128, 0))
       
       # Vectors  
       position = server.gui.add_vector3("Position", initial_value=(0.0, 1.0, 2.0))
       
       # Actions
       button = server.gui.add_button("Execute")
       reset_button = server.gui.add_button("Reset All")

**Handling Events**

.. code-block:: python

   # Respond to slider changes
   @slider.on_update
   def update_slider() -> None:
       print(f"Slider value: {slider.value}")
       # Update your visualization here
   
   # Handle button clicks
   @button.on_click
   def execute_action() -> None:
       print("Executing action...")
       # Perform some operation
   
   @reset_button.on_click
   def reset_all() -> None:
       slider.value = 50
       checkbox.value = True
       color_picker.value = (255, 128, 0)

Organization & Layout
---------------------

Create well-organized interfaces that scale with complexity.

**Folder Hierarchy**

.. code-block:: python

   server = viser.ViserServer()
   
   # Main categories
   with server.gui.add_folder("Visualization"):
       with server.gui.add_folder("Objects"):
           show_points = server.gui.add_checkbox("Show Points", initial_value=True)
           point_size = server.gui.add_slider("Point Size", min=0.01, max=0.1, step=0.01)
           point_color = server.gui.add_rgb("Point Color", initial_value=(0, 255, 0))
       
       with server.gui.add_folder("Lighting"):
           ambient_strength = server.gui.add_slider("Ambient", min=0.0, max=1.0, step=0.1, initial_value=0.3)
           directional_strength = server.gui.add_slider("Directional", min=0.0, max=2.0, step=0.1, initial_value=1.0)
   
   with server.gui.add_folder("Camera"):
       camera_speed = server.gui.add_slider("Movement Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
       field_of_view = server.gui.add_slider("Field of View", min=30, max=120, step=5, initial_value=75)
   
   with server.gui.add_folder("Data"):
       with server.gui.add_folder("Loading"):
           file_path = server.gui.add_text("File Path", initial_value="")
           load_button = server.gui.add_button("Load Data")
       
       with server.gui.add_folder("Processing"):
           filter_threshold = server.gui.add_slider("Filter Threshold", min=0.0, max=1.0, step=0.01)
           apply_filter = server.gui.add_checkbox("Apply Filter", initial_value=False)

**Responsive Layout**

.. code-block:: python

   # Create dynamic interface that adapts to data
   server = viser.ViserServer()
   
   # Main data controls
   with server.gui.add_folder("Dataset") as dataset_folder:
       dataset_type = server.gui.add_dropdown(
           "Type", 
           options=["Point Cloud", "Mesh", "Trajectory", "Volume"]
       )
   
   # Dynamic controls based on dataset type
   current_controls = {}
   
   @dataset_type.on_update
   def update_controls() -> None:
       # Remove old controls
       for control in current_controls.values():
           control.remove()
       current_controls.clear()
       
       # Add type-specific controls
       if dataset_type.value == "Point Cloud":
           current_controls["size"] = server.gui.add_slider(
               "Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
           )
           current_controls["decimation"] = server.gui.add_slider(
               "Decimation", min=1, max=100, step=1, initial_value=1
           )
       
       elif dataset_type.value == "Mesh":
           current_controls["wireframe"] = server.gui.add_checkbox(
               "Wireframe", initial_value=False
           )
           current_controls["opacity"] = server.gui.add_slider(
               "Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0
           )
       
       elif dataset_type.value == "Trajectory":
           current_controls["line_width"] = server.gui.add_slider(
               "Line Width", min=1.0, max=10.0, step=0.5, initial_value=2.0
           )
           current_controls["show_poses"] = server.gui.add_checkbox(
               "Show Poses", initial_value=True
           )

Advanced Interactions
---------------------

Build sophisticated interfaces with modals, tabs, and custom components.

**Modal Dialogs**

.. code-block:: python

   import viser
   
   server = viser.ViserServer()
   
   # Main interface
   open_settings = server.gui.add_button("Open Settings")
   
   @open_settings.on_click
   def show_settings() -> None:
       with server.gui.add_modal("Settings") as modal:
           # Settings content
           with server.gui.add_folder("Rendering"):
               quality = server.gui.add_dropdown(
                   "Quality", 
                   options=["Low", "Medium", "High", "Ultra"]
               )
               anti_aliasing = server.gui.add_checkbox("Anti-aliasing", initial_value=True)
           
           with server.gui.add_folder("Performance"):
               max_points = server.gui.add_number("Max Points", initial_value=100000)
               update_rate = server.gui.add_slider("Update Rate (Hz)", min=1, max=60, step=1, initial_value=30)
           
           # Modal actions
           with server.gui.add_folder("Actions"):
               apply_button = server.gui.add_button("Apply")
               cancel_button = server.gui.add_button("Cancel")
           
           @apply_button.on_click
           def apply_settings() -> None:
               # Apply the settings
               print(f"Applied: Quality={quality.value}, AA={anti_aliasing.value}")
               modal.close()
           
           @cancel_button.on_click
           def cancel_settings() -> None:
               modal.close()

**Progress Indicators**

.. code-block:: python

   import time
   import threading
   import viser
   
   server = viser.ViserServer()
   
   # Progress bar for long operations
   process_button = server.gui.add_button("Start Processing")
   progress_bar = server.gui.add_progress_bar(0, animated=False)
   status_text = server.gui.add_text("Status", "Ready")
   
   @process_button.on_click
   def start_processing() -> None:
       def background_task() -> None:
           process_button.disabled = True
           
           for i in range(101):
               progress_bar.value = i
               status_text.value = f"Processing... {i}%"
               time.sleep(0.05)  # Simulate work
           
           status_text.value = "Complete!"
           process_button.disabled = False
       
       # Run in background thread
       threading.Thread(target=background_task, daemon=True).start()

**Multi-Step Workflows**

.. code-block:: python

   import viser
   
   server = viser.ViserServer()
   
   # Wizard-style interface
   current_step = 0
   steps = ["Data Selection", "Preprocessing", "Visualization", "Export"]
   
   # Step indicator
   step_text = server.gui.add_text("Current Step", f"Step {current_step + 1}: {steps[current_step]}")
   
   # Navigation
   with server.gui.add_folder("Navigation"):
       prev_button = server.gui.add_button("Previous")
       next_button = server.gui.add_button("Next")
   
   # Dynamic content area
   content_folder = server.gui.add_folder("Content")
   current_controls = []
   
   def show_step(step: int) -> None:
       # Clear current content
       for control in current_controls:
           control.remove()
       current_controls.clear()
       
       step_text.value = f"Step {step + 1}: {steps[step]}"
       
       if step == 0:  # Data Selection
           current_controls.append(
               server.gui.add_text("Data Path", "Enter path to your data file")
           )
           current_controls.append(
               server.gui.add_dropdown("Data Type", options=["CSV", "PLY", "OBJ"])
           )
       
       elif step == 1:  # Preprocessing
           current_controls.append(
               server.gui.add_checkbox("Remove Outliers", initial_value=True)
           )
           current_controls.append(
               server.gui.add_slider("Noise Threshold", min=0.0, max=1.0, step=0.01)
           )
       
       elif step == 2:  # Visualization
           current_controls.append(
               server.gui.add_rgb("Color Scheme", initial_value=(100, 150, 255))
           )
           current_controls.append(
               server.gui.add_checkbox("Show Grid", initial_value=True)
           )
       
       elif step == 3:  # Export
           current_controls.append(
               server.gui.add_dropdown("Format", options=["PNG", "PDF", "SVG"])
           )
           current_controls.append(
               server.gui.add_slider("Resolution", min=720, max=4320, step=360, initial_value=1080)
           )
       
       # Update button states
       prev_button.disabled = (step == 0)
       next_button.disabled = (step == len(steps) - 1)
   
   @prev_button.on_click
   def go_previous() -> None:
       global current_step
       if current_step > 0:
           current_step -= 1
           show_step(current_step)
   
   @next_button.on_click
   def go_next() -> None:
       global current_step
       if current_step < len(steps) - 1:
           current_step += 1
           show_step(current_step)
   
   # Initialize first step
   show_step(0)

Real-time Updates
-----------------

Create responsive interfaces that update visualizations in real-time.

**Coordinated Updates**

.. code-block:: python

   import numpy as np
   import viser
   
   server = viser.ViserServer()
   
   # Create initial objects
   sphere = server.scene.add_icosphere("sphere", radius=0.5, color=(255, 0, 0))
   
   # Controls that affect multiple properties
   with server.gui.add_folder("Object Properties"):
       size_slider = server.gui.add_slider("Size", min=0.1, max=2.0, step=0.1, initial_value=0.5)
       color_picker = server.gui.add_rgb("Color", initial_value=(255, 0, 0))
       position_vector = server.gui.add_vector3("Position", initial_value=(0.0, 0.0, 0.0))
       visible_checkbox = server.gui.add_checkbox("Visible", initial_value=True)
   
   # Coordinate all updates
   def update_sphere() -> None:
       sphere.radius = size_slider.value
       sphere.color = color_picker.value
       sphere.position = position_vector.value
       sphere.visible = visible_checkbox.value
   
   # Connect all controls to the update function
   size_slider.on_update(update_sphere)
   color_picker.on_update(update_sphere)
   position_vector.on_update(update_sphere)
   visible_checkbox.on_update(update_sphere)

**Computed Properties**

.. code-block:: python

   # Create dependent controls that compute values from others
   with server.gui.add_folder("Circle"):
       radius_slider = server.gui.add_slider("Radius", min=0.1, max=5.0, step=0.1, initial_value=1.0)
       
       # Computed readonly displays
       circumference_display = server.gui.add_text("Circumference", "6.28", disabled=True)
       area_display = server.gui.add_text("Area", "3.14", disabled=True)
   
   @radius_slider.on_update
   def update_circle_properties() -> None:
       radius = radius_slider.value
       circumference = 2 * np.pi * radius
       area = np.pi * radius ** 2
       
       circumference_display.value = f"{circumference:.2f}"
       area_display.value = f"{area:.2f}"
       
       # Update visualization
       circle_object.radius = radius

**State Management**

.. code-block:: python

   # Manage complex application state
   class AppState:
       def __init__(self):
           self.data_loaded = False
           self.processing_active = False
           self.current_dataset = None
           self.view_mode = "3D"
           
       def update_ui(self) -> None:
           """Update UI elements based on current state."""
           # Enable/disable controls based on state
           load_button.disabled = self.processing_active
           process_button.disabled = not self.data_loaded or self.processing_active
           export_button.disabled = not self.data_loaded
           
           # Update status display
           if self.processing_active:
               status_text.value = "Processing..."
           elif self.data_loaded:
               status_text.value = f"Loaded: {len(self.current_dataset)} points"
           else:
               status_text.value = "No data loaded"
   
   app_state = AppState()
   
   # UI controls
   load_button = server.gui.add_button("Load Data")
   process_button = server.gui.add_button("Process")
   export_button = server.gui.add_button("Export")
   status_text = server.gui.add_text("Status", "No data loaded", disabled=True)
   
   @load_button.on_click
   def load_data() -> None:
       app_state.current_dataset = load_your_data()  # Your data loading function
       app_state.data_loaded = True
       app_state.update_ui()
   
   @process_button.on_click
   def process_data() -> None:
       app_state.processing_active = True
       app_state.update_ui()
       
       # Process in background
       def background_process() -> None:
           process_your_data(app_state.current_dataset)  # Your processing function
           app_state.processing_active = False
           app_state.update_ui()
       
       threading.Thread(target=background_process, daemon=True).start()
   
   # Initialize UI state
   app_state.update_ui()

Best Practices
--------------

**1. Logical Organization**

.. code-block:: python

   # Group related controls together
   with server.gui.add_folder("Scene Objects"):
       with server.gui.add_folder("Point Cloud"):
           # All point cloud controls here
           pass
       
       with server.gui.add_folder("Meshes"):
           # All mesh controls here  
           pass
   
   with server.gui.add_folder("Rendering"):
       # All rendering controls here
       pass

**2. Provide Visual Feedback**

.. code-block:: python

   # Show what controls do
   @color_picker.on_update
   def update_color() -> None:
       sphere.color = color_picker.value
       # Also update a preview or status
       status_text.value = f"Color: RGB{color_picker.value}"

**3. Use Appropriate Controls**

.. code-block:: python

   # Choose the right control for the data type
   
   # For continuous values
   slider = server.gui.add_slider("Opacity", min=0.0, max=1.0, step=0.01)
   
   # For discrete choices
   mode = server.gui.add_dropdown("Mode", options=["View", "Edit", "Analyze"])
   
   # For on/off states
   enabled = server.gui.add_checkbox("Enable Feature")
   
   # For precise numeric input
   threshold = server.gui.add_number("Threshold", step=0.001)

**4. Handle Edge Cases**

.. code-block:: python

   @file_path.on_update
   def validate_file_path() -> None:
       import os
       if os.path.exists(file_path.value):
           load_button.disabled = False
           status_text.value = "File found"
           status_text.color = (0, 255, 0)  # Green
       else:
           load_button.disabled = True
           status_text.value = "File not found"
           status_text.color = (255, 0, 0)  # Red

**5. Performance Considerations**

.. code-block:: python

   import time
   
   # Debounce rapid updates
   last_update_time = 0
   
   @expensive_slider.on_update
   def debounced_update() -> None:
       global last_update_time
       current_time = time.time()
       
       # Only update if enough time has passed
       if current_time - last_update_time > 0.1:  # 100ms debounce
           expensive_computation()
           last_update_time = current_time

Common Patterns
---------------

**Configuration Panels**

.. code-block:: python

   # Reusable configuration pattern
   def create_point_cloud_controls(name: str, initial_values=None):
       if initial_values is None:
           initial_values = {"size": 0.01, "color": (255, 255, 255), "visible": True}
       
       with server.gui.add_folder(name):
           size = server.gui.add_slider("Size", min=0.001, max=0.1, step=0.001, initial_value=initial_values["size"])
           color = server.gui.add_rgb("Color", initial_value=initial_values["color"])
           visible = server.gui.add_checkbox("Visible", initial_value=initial_values["visible"])
       
       return {"size": size, "color": color, "visible": visible}
   
   # Use for multiple point clouds
   lidar_controls = create_point_cloud_controls("LiDAR", {"size": 0.02, "color": (0, 255, 0)})
   camera_controls = create_point_cloud_controls("Camera Points", {"size": 0.01, "color": (255, 0, 0)})

**Data Binding**

.. code-block:: python

   # Automatically sync UI with data structures
   class VisualizationConfig:
       def __init__(self):
           self.point_size = 0.01
           self.show_grid = True
           self.background_color = (50, 50, 50)
       
       def create_ui(self, server) -> None:
           # Create UI controls
           self._size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=self.point_size)
           self._grid_checkbox = server.gui.add_checkbox("Show Grid", initial_value=self.show_grid)
           self._bg_color = server.gui.add_rgb("Background", initial_value=self.background_color)
           
           # Bind to properties
           self._size_slider.on_update(lambda: setattr(self, 'point_size', self._size_slider.value))
           self._grid_checkbox.on_update(lambda: setattr(self, 'show_grid', self._grid_checkbox.value))
           self._bg_color.on_update(lambda: setattr(self, 'background_color', self._bg_color.value))
   
   config = VisualizationConfig()
   config.create_ui(server)

Next Steps
----------

- **See examples**: :doc:`../examples/02_gui_index`
- **Add interaction**: :doc:`handling_interaction`
- **Optimize performance**: :doc:`performance`
- **Domain-specific UIs**: :doc:`domain_specific`