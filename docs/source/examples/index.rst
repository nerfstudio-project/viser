Examples
========

Learn viser through hands-on examples organized by functionality and complexity.

.. toctree::
   :hidden:
   :caption: Example Source Code
   
   generated/index

.. note::
   All examples are runnable Python scripts in the ``examples_dev/`` directory. Each file contains extensive comments explaining the concepts and implementation details.

Quick Start
-----------

New to viser? Start with the **Hello World** example::

    python examples_dev/00_hello_world.py

Then follow this learning path:

1. **🎯 Scene Basics** → Master 3D visualization fundamentals
2. **🎛️ GUI Controls** → Build interactive interfaces  
3. **🖱️ User Interaction** → Handle clicks and events
4. **🚀 Complete Apps** → See real-world applications

Getting the Code
----------------

**Download all examples:**

.. code-block:: bash

   git clone https://github.com/nerfstudio-project/viser.git
   cd viser/examples_dev

**Install dependencies:**

.. code-block:: bash

   pip install viser[examples]

**Run any example:**

.. code-block:: bash

   python 00_hello_world.py
   python 01_scene/00_coordinate_frames.py
   python 02_gui/00_basic_controls.py

Then open http://localhost:8080 in your browser!

Example Gallery
---------------

.. include:: _example_gallery.rst

Example Categories
------------------

**🎯 Scene Fundamentals** (``examples_dev/01_scene/``)
   Learn the core concepts: coordinate systems, 3D objects, cameras, and lighting. Essential for understanding how viser works.

   Examples include:
   - ``00_coordinate_frames.py`` - Visualize 3D coordinate systems
   - ``01_images.py`` - Display images in 3D space
   - ``02_meshes.py`` - Load and display 3D meshes
   - ``03_lines.py`` - Draw lines and paths
   - ``04_meshes_batched.py`` - Efficiently render multiple meshes
   - ``05_camera_poses.py`` - Visualize camera positions
   - ``06_camera_commands.py`` - Control camera programmatically
   - ``07_lighting.py`` - Configure scene lighting
   - ``08_background_composite.py`` - Composite backgrounds
   - ``09_set_up_direction.py`` - Change coordinate conventions

**🎛️ GUI Controls** (``examples_dev/02_gui/``)
   Build user interfaces with sliders, buttons, and custom layouts. Make your visualizations interactive and user-friendly.

   Examples include:
   - ``00_basic_controls.py`` - Basic GUI elements
   - ``01_callbacks.py`` - Handle user interactions
   - ``02_layouts.py`` - Organize GUI layouts
   - ``03_markdown.py`` - Rich text and formatting
   - ``04_modals.py`` - Modal dialogs
   - ``05_theming.py`` - Customize appearance
   - ``06_gui_in_scene.py`` - 3D GUI elements
   - ``07_notifications.py`` - User notifications
   - ``08_plotly_integration.py`` - Interactive plots
   - ``09_plots_as_images.py`` - Display plots as images

**🖱️ User Interaction** (``examples_dev/03_interaction/``)
   Handle mouse clicks, scene picking, and real-time user input. Create responsive, interactive applications.

   Examples include:
   - ``00_click_meshes.py`` - Handle mesh clicks
   - ``01_scene_pointer.py`` - Track mouse position in 3D
   - ``02_get_renders.py`` - Capture rendered images
   - ``03_games.py`` - Interactive games and demos

**🚀 Complete Applications** (``examples_dev/04_demos/``)
   Real-world examples integrating external tools and datasets. See how viser works with robotics, computer vision, and 3D reconstruction.

   Examples include:
   - ``00_record3d_visualizer.py`` - Visualize iPhone captures
   - ``02_colmap_visualizer.py`` - COLMAP reconstruction viewer
   - ``03_urdf_visualizer.py`` - Robot model visualization
   - ``04_smpl_visualizer.py`` - Human body models
   - ``05_smpl_skinned.py`` - Skinned mesh animation

Tips for Learning
------------------

**1. Start Simple**
   Begin with ``00_hello_world.py`` to understand the basics, then progress through each category starting with ``01_scene/``.

**2. Read the Code**
   Each example file is self-contained with extensive comments. The code itself is the best documentation!

**3. Experiment**
   Modify the examples! Change colors, positions, add new objects. The best way to learn is by doing.

**4. Combine Concepts**
   Once you understand the basics, combine techniques from different examples to build your own applications.

**5. Use the Documentation**
   Reference the :doc:`../user_guides/index` for task-oriented tutorials and :doc:`../server` for complete API documentation.

Finding Examples
----------------

Browse the examples directory to discover all available examples:

.. code-block:: bash

   # List all categories
   ls examples_dev/
   
   # List scene examples
   ls examples_dev/01_scene/
   
   # Find examples by keyword
   grep -r "point_cloud" examples_dev/
   grep -r "slider" examples_dev/

Need Help?
----------

- **🔧 Troubleshooting**: See :doc:`../troubleshooting` for common issues
- **💬 Community**: Ask questions on GitHub Issues
- **📚 API Reference**: Check :doc:`../server` for detailed documentation
- **🎯 User Guides**: Browse :doc:`../user_guides/index` for task-oriented tutorials

Ready to start? Run ``python examples_dev/00_hello_world.py``!