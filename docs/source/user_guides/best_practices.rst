Best Practices
==============

Guidelines for building robust, maintainable, and performant viser applications.

Project Organization
--------------------

**Structure your visualization code clearly:**

.. code-block:: text

   my_project/
   ├── main.py              # Entry point
   ├── visualization/
   │   ├── __init__.py
   │   ├── scene.py         # Scene setup and management
   │   ├── gui.py           # GUI controls and layout
   │   └── data.py          # Data loading and processing
   ├── config/
   │   └── settings.py      # Configuration constants
   ├── assets/
   │   ├── models/
   │   └── textures/
   └── requirements.txt

**Example main.py structure:**

.. code-block:: python

   import viser
   from visualization.scene import SceneManager
   from visualization.gui import GUIManager
   from visualization.data import DataLoader
   
   def main():
       server = viser.ViserServer()
       
       # Initialize components
       data_loader = DataLoader()
       scene_manager = SceneManager(server.scene)
       gui_manager = GUIManager(server.gui, scene_manager)
       
       # Load and visualize data
       data = data_loader.load("path/to/data")
       scene_manager.visualize(data)
       
       # Start GUI
       gui_manager.setup_controls()
       
       print("Visit http://localhost:8080")
       while True:
           pass
   
   if __name__ == "__main__":
       main()

Code Organization
-----------------

**1. Separate Concerns**

Keep visualization logic, data processing, and GUI code in separate modules:

.. code-block:: python

   # scene.py - Scene management
   class SceneManager:
       def __init__(self, scene_api):
           self.scene = scene_api
           self.objects = {}
       
       def add_point_cloud(self, name, points, colors=None):
           if colors is None:
               colors = self._generate_height_colors(points)
           
           handle = self.scene.add_point_cloud(name, points=points, colors=colors)
           self.objects[name] = handle
           return handle
       
       def _generate_height_colors(self, points):
           # Color by height
           z_vals = points[:, 2]
           norm_z = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
           return np.column_stack([norm_z * 255, np.zeros_like(norm_z), (1-norm_z) * 255]).astype(np.uint8)

.. code-block:: python

   # gui.py - GUI management
   class GUIManager:
       def __init__(self, gui_api, scene_manager):
           self.gui = gui_api
           self.scene_manager = scene_manager
           self.controls = {}
       
       def setup_controls(self):
           with self.gui.add_folder("Data Visualization"):
               self.controls["point_size"] = self.gui.add_slider(
                   "Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
               )
               self.controls["color_mode"] = self.gui.add_dropdown(
                   "Color Mode", options=["Height", "Random", "Solid"]
               )
           
           # Connect events
           self.controls["point_size"].on_update(self._update_point_size)
           self.controls["color_mode"].on_update(self._update_colors)

**2. Use Configuration Files**

.. code-block:: python

   # config/settings.py
   import dataclasses
   from typing import Tuple
   
   @dataclasses.dataclass
   class VisualizationConfig:
       # Server settings
       host: str = "localhost"
       port: int = 8080
       
       # Rendering settings
       default_point_size: float = 0.01
       max_points: int = 100000
       
       # Colors (RGB tuples)
       background_color: Tuple[int, int, int] = (50, 50, 50)
       default_object_color: Tuple[int, int, int] = (150, 150, 150)
       
       # Performance
       update_rate_hz: float = 30.0
       decimation_factor: int = 1

**3. Error Handling**

Implement robust error handling throughout your application:

.. code-block:: python

   import logging
   
   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   class DataLoader:
       def load_point_cloud(self, file_path):
           try:
               if file_path.suffix == '.ply':
                   return self._load_ply(file_path)
               elif file_path.suffix == '.pcd':
                   return self._load_pcd(file_path)
               else:
                   raise ValueError(f"Unsupported file format: {file_path.suffix}")
           
           except FileNotFoundError:
               logger.error(f"File not found: {file_path}")
               return self._create_fallback_data()
           
           except Exception as e:
               logger.error(f"Error loading {file_path}: {e}")
               return self._create_fallback_data()
       
       def _create_fallback_data(self):
           """Create simple fallback data when loading fails."""
           points = np.random.randn(1000, 3)
           colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
           return {"points": points, "colors": colors}

Performance Best Practices
---------------------------

**1. Data Optimization**

.. code-block:: python

   def optimize_point_cloud(points, colors=None, max_points=100000):
       """Optimize point cloud for visualization."""
       
       # Decimate if too many points
       if len(points) > max_points:
           step = len(points) // max_points
           points = points[::step]
           if colors is not None:
               colors = colors[::step]
       
       # Use appropriate data types
       points = points.astype(np.float32)  # Sufficient precision
       if colors is not None:
           colors = colors.astype(np.uint8)  # 0-255 color range
       
       return points, colors

**2. Efficient Updates**

.. code-block:: python

   import time
   from collections import deque
   
   class PerformanceManager:
       def __init__(self, target_fps=30):
           self.target_fps = target_fps
           self.frame_times = deque(maxlen=30)
           self.last_update = 0
       
       def should_update(self):
           """Throttle updates to maintain target FPS."""
           current_time = time.time()
           time_since_update = current_time - self.last_update
           
           if time_since_update >= 1.0 / self.target_fps:
               self.last_update = current_time
               return True
           return False
       
       def record_frame_time(self, frame_time):
           self.frame_times.append(frame_time)
       
       def get_fps(self):
           if not self.frame_times:
               return 0
           return 1.0 / np.mean(self.frame_times)

**3. Memory Management**

.. code-block:: python

   class SceneManager:
       def __init__(self, scene_api):
           self.scene = scene_api
           self.objects = {}
           self._max_objects = 1000
       
       def add_object(self, name, **kwargs):
           # Remove old objects if we have too many
           if len(self.objects) >= self._max_objects:
               self._cleanup_old_objects()
           
           handle = self.scene.add_icosphere(name, **kwargs)
           self.objects[name] = {
               'handle': handle,
               'created_at': time.time()
           }
           return handle
       
       def _cleanup_old_objects(self):
           """Remove oldest objects to free memory."""
           # Sort by creation time
           sorted_objects = sorted(
               self.objects.items(),
               key=lambda x: x[1]['created_at']
           )
           
           # Remove oldest 25%
           num_to_remove = len(sorted_objects) // 4
           for name, obj_info in sorted_objects[:num_to_remove]:
               obj_info['handle'].remove()
               del self.objects[name]

GUI Design Principles
---------------------

**1. Progressive Disclosure**

Start with essential controls visible, hide advanced options:

.. code-block:: python

   def setup_gui(self):
       # Essential controls always visible
       with self.gui.add_folder("Basic Controls"):
           self.point_size = self.gui.add_slider("Point Size", min=0.001, max=0.1)
           self.show_data = self.gui.add_checkbox("Show Data", initial_value=True)
       
       # Advanced controls in collapsed folder
       with self.gui.add_folder("Advanced", expanded=False):
           self.decimation = self.gui.add_slider("Decimation", min=1, max=100)
           self.color_scheme = self.gui.add_dropdown("Color Scheme", options=["Default", "Height", "Custom"])
       
       # Expert controls in modal
       self.settings_button = self.gui.add_button("Advanced Settings...")
       
       @self.settings_button.on_click
       def show_settings():
           with self.gui.add_modal("Settings") as modal:
               # Complex settings here
               pass

**2. Immediate Feedback**

Provide instant visual feedback for user actions:

.. code-block:: python

   @self.point_size_slider.on_update
   def update_point_size():
       # Update visualization immediately
       for obj in self.point_cloud_objects:
           obj.point_size = self.point_size_slider.value
       
       # Show current value
       self.status_text.value = f"Point size: {self.point_size_slider.value:.3f}"

**3. Consistent Layouts**

Use consistent patterns throughout your interface:

.. code-block:: python

   def create_object_controls(self, object_name):
       """Reusable pattern for object controls."""
       with self.gui.add_folder(f"{object_name} Controls"):
           controls = {
               'visible': self.gui.add_checkbox("Visible", initial_value=True),
               'color': self.gui.add_rgb("Color", initial_value=(255, 255, 255)),
               'opacity': self.gui.add_slider("Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0)
           }
           
           # Standard reset button
           reset_button = self.gui.add_button("Reset to Default")
           
           @reset_button.on_click
           def reset_controls():
               controls['visible'].value = True
               controls['color'].value = (255, 255, 255)
               controls['opacity'].value = 1.0
           
           return controls

Data Management
---------------

**1. Lazy Loading**

Load data only when needed:

.. code-block:: python

   class DataManager:
       def __init__(self):
           self._cache = {}
           self._loaded_datasets = set()
       
       def get_dataset(self, name):
           if name not in self._cache:
               self._cache[name] = self._load_dataset(name)
               self._loaded_datasets.add(name)
           return self._cache[name]
       
       def unload_dataset(self, name):
           if name in self._cache:
               del self._cache[name]
               self._loaded_datasets.discard(name)

**2. Data Validation**

Validate data before visualization:

.. code-block:: python

   def validate_point_cloud(points, colors=None):
       """Validate point cloud data before visualization."""
       
       # Check shape
       if not isinstance(points, np.ndarray):
           raise TypeError("Points must be numpy array")
       
       if points.ndim != 2 or points.shape[1] != 3:
           raise ValueError(f"Points must be (N, 3) array, got {points.shape}")
       
       # Check for invalid values
       if not np.isfinite(points).all():
           raise ValueError("Points contain invalid values (inf/nan)")
       
       # Validate colors if provided
       if colors is not None:
           if colors.shape != (len(points), 3):
               raise ValueError(f"Colors shape {colors.shape} doesn't match points {points.shape}")
           
           if colors.dtype != np.uint8:
               if colors.max() <= 1.0:
                   colors = (colors * 255).astype(np.uint8)
               else:
                   colors = colors.astype(np.uint8)
       
       return points, colors

**3. Configuration Management**

Make your application configurable:

.. code-block:: python

   import json
   from pathlib import Path
   
   class ConfigManager:
       def __init__(self, config_path="config.json"):
           self.config_path = Path(config_path)
           self.config = self.load_config()
       
       def load_config(self):
           if self.config_path.exists():
               with open(self.config_path) as f:
                   return json.load(f)
           else:
               return self.get_default_config()
       
       def save_config(self):
           with open(self.config_path, 'w') as f:
               json.dump(self.config, f, indent=2)
       
       def get_default_config(self):
           return {
               "visualization": {
                   "point_size": 0.01,
                   "max_points": 100000,
                   "color_scheme": "height"
               },
               "server": {
                   "port": 8080,
                   "host": "localhost"
               }
           }

Testing and Debugging
----------------------

**1. Unit Tests for Data Processing**

.. code-block:: python

   import unittest
   import numpy as np
   
   class TestDataProcessing(unittest.TestCase):
       def test_point_cloud_validation(self):
           # Valid data
           points = np.random.randn(100, 3)
           colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
           
           validated_points, validated_colors = validate_point_cloud(points, colors)
           self.assertEqual(validated_points.shape, (100, 3))
           self.assertEqual(validated_colors.shape, (100, 3))
       
       def test_invalid_point_cloud(self):
           # Invalid shape
           points = np.random.randn(100, 2)  # Wrong dimension
           
           with self.assertRaises(ValueError):
               validate_point_cloud(points)

**2. Debug Information**

Add debug information to your GUI:

.. code-block:: python

   def setup_debug_info(self):
       with self.gui.add_folder("Debug Info", expanded=False):
           self.debug_info = {
               'fps': self.gui.add_text("FPS", "0.0", disabled=True),
               'memory': self.gui.add_text("Memory (MB)", "0", disabled=True),
               'objects': self.gui.add_text("Scene Objects", "0", disabled=True),
               'points': self.gui.add_text("Total Points", "0", disabled=True)
           }
   
   def update_debug_info(self):
       self.debug_info['fps'].value = f"{self.performance_manager.get_fps():.1f}"
       self.debug_info['memory'].value = f"{self.get_memory_usage():.1f}"
       self.debug_info['objects'].value = str(len(self.scene_manager.objects))
       self.debug_info['points'].value = str(self.get_total_points())

**3. Logging**

Use structured logging for debugging:

.. code-block:: python

   import logging
   
   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   logger = logging.getLogger(__name__)
   
   class SceneManager:
       def add_object(self, name, **kwargs):
           logger.info(f"Adding object: {name}")
           
           try:
               handle = self.scene.add_icosphere(name, **kwargs)
               logger.debug(f"Successfully added {name} with handle {handle}")
               return handle
           
           except Exception as e:
               logger.error(f"Failed to add object {name}: {e}")
               raise

Deployment Considerations
-------------------------

**1. Production Configuration**

.. code-block:: python

   import os
   
   # Use environment variables for production settings
   class ProductionConfig:
       HOST = os.getenv("VISER_HOST", "0.0.0.0")  # Allow external connections
       PORT = int(os.getenv("VISER_PORT", "8080"))
       DEBUG = os.getenv("VISER_DEBUG", "False").lower() == "true"
       MAX_POINTS = int(os.getenv("VISER_MAX_POINTS", "100000"))

**2. Resource Limits**

.. code-block:: python

   class ResourceManager:
       def __init__(self, max_memory_mb=1000, max_objects=1000):
           self.max_memory_mb = max_memory_mb
           self.max_objects = max_objects
       
       def check_limits(self):
           memory_usage = self.get_memory_usage()
           object_count = len(self.scene_manager.objects)
           
           if memory_usage > self.max_memory_mb:
               logger.warning(f"Memory usage ({memory_usage:.1f} MB) exceeds limit")
               self.cleanup_memory()
           
           if object_count > self.max_objects:
               logger.warning(f"Object count ({object_count}) exceeds limit")
               self.cleanup_objects()

**3. Error Recovery**

.. code-block:: python

   def robust_visualization_loop(self):
       """Main loop with error recovery."""
       error_count = 0
       max_errors = 10
       
       while error_count < max_errors:
           try:
               self.update_visualization()
               error_count = 0  # Reset on success
               
           except Exception as e:
               error_count += 1
               logger.error(f"Visualization error ({error_count}/{max_errors}): {e}")
               
               if error_count >= max_errors:
                   logger.critical("Too many errors, shutting down")
                   break
               
               # Try to recover
               self.reset_scene()
               time.sleep(1)  # Brief pause before retry

Summary Checklist
-----------------

**Code Organization:**
- ✅ Separate visualization, GUI, and data processing code
- ✅ Use configuration files for settings
- ✅ Implement proper error handling and logging
- ✅ Write tests for data processing functions

**Performance:**
- ✅ Optimize data types (float32, uint8)
- ✅ Implement point cloud decimation for large datasets
- ✅ Use batching for many similar objects
- ✅ Add frame rate limiting and performance monitoring

**User Experience:**
- ✅ Progressive disclosure of controls
- ✅ Immediate visual feedback
- ✅ Consistent UI patterns
- ✅ Debug information for development

**Production:**
- ✅ Environment-based configuration
- ✅ Resource limits and monitoring
- ✅ Error recovery mechanisms
- ✅ Proper logging for debugging

Following these practices will help you build robust, maintainable viser applications that perform well and provide a great user experience.