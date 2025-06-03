Scene Fundamentals
==================

Master the core concepts of 3D visualization with viser. These examples teach you how to create, position, and render 3D objects.

.. note::
   **Start here if you're new to viser!** These examples build the foundation for everything else.

Learning Objectives
-------------------

By working through these examples, you'll learn:

- ✅ **Coordinate Systems** - How 3D space works in viser
- ✅ **3D Objects** - Adding spheres, boxes, meshes, and point clouds  
- ✅ **Positioning & Rotation** - Placing objects in 3D space
- ✅ **Camera Control** - Programmatic camera movement
- ✅ **Lighting & Materials** - Making objects look realistic
- ✅ **Performance** - Efficiently rendering many objects

Available Examples
------------------

Browse the ``examples_dev/01_scene/`` directory to find these examples:

.. raw:: html

   <style>
   .example-gallery {
       display: grid;
       grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
       gap: 20px;
       margin: 20px 0;
   }
   .example-card {
       border: 1px solid #ddd;
       border-radius: 8px;
       overflow: hidden;
       transition: transform 0.2s;
       background: white;
       text-decoration: none;
       display: block;
       color: inherit;
   }
   .example-card:hover {
       transform: translateY(-4px);
       box-shadow: 0 4px 12px rgba(0,0,0,0.15);
   }
   .example-card img {
       width: 100%;
       height: 180px;
       object-fit: cover;
   }
   .example-card-content {
       padding: 15px;
   }
   .example-card h4 {
       margin: 0 0 8px 0;
       font-size: 16px;
       font-family: monospace;
   }
   .example-card p {
       margin: 0;
       color: #666;
       font-size: 14px;
   }
   </style>

   <div class="example-gallery">
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/00_coordinate_frames.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_00_coordinate_frames.png" alt="Coordinate Frames">
           <div class="example-card-content">
               <h4>00_coordinate_frames.py</h4>
               <p>🟢 Understanding 3D coordinate systems</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/01_images.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_01_images.png" alt="Images">
           <div class="example-card-content">
               <h4>01_images.py</h4>
               <p>🟢 Display images in 3D space</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/02_meshes.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_02_meshes.png" alt="Meshes">
           <div class="example-card-content">
               <h4>02_meshes.py</h4>
               <p>🟡 Load and display 3D meshes</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/03_lines.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_03_lines.png" alt="Lines">
           <div class="example-card-content">
               <h4>03_lines.py</h4>
               <p>🟢 Draw lines and paths</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/04_meshes_batched.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_04_meshes_batched.png" alt="Batched Meshes">
           <div class="example-card-content">
               <h4>04_meshes_batched.py</h4>
               <p>🔴 Efficiently render many meshes</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/05_camera_poses.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_05_camera_poses.png" alt="Camera Poses">
           <div class="example-card-content">
               <h4>05_camera_poses.py</h4>
               <p>🟡 Control camera position</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/06_camera_commands.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_06_camera_commands.png" alt="Camera Commands">
           <div class="example-card-content">
               <h4>06_camera_commands.py</h4>
               <p>🔴 Programmatic camera animation</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/07_lighting.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_07_lighting.png" alt="Lighting">
           <div class="example-card-content">
               <h4>07_lighting.py</h4>
               <p>🔴 Advanced lighting and materials</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/08_background_composite.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_08_background_composite.png" alt="Background">
           <div class="example-card-content">
               <h4>08_background_composite.py</h4>
               <p>🔴 Custom backgrounds</p>
           </div>
       </a>
       <a href="https://github.com/nerfstudio-project/viser/blob/main/examples_dev/01_scene/09_set_up_direction.py" class="example-card" target="_blank">
           <img src="../../_static/examples/01_scene_09_set_up_direction.png" alt="Up Direction">
           <div class="example-card-content">
               <h4>09_set_up_direction.py</h4>
               <p>🟡 Configure scene orientation</p>
           </div>
       </a>
   </div>

Running the Examples
---------------------

.. code-block:: bash

   # Navigate to examples directory
   cd viser/examples_dev
   
   # Run any scene example
   python 01_scene/00_coordinate_frames.py
   python 01_scene/02_meshes.py
   
   # Open browser to http://localhost:8080

Key Concepts
------------

**Coordinate Systems**
   Viser uses right-handed coordinates with Y-up. Understanding this is crucial for positioning objects correctly.

**Scene Graph**
   Objects are organized hierarchically. Parent transformations affect all children.

**Real-time Updates**
   You can modify object properties (position, color, visibility) while the visualization is running.

**Performance**
   For large datasets, use batching and level-of-detail techniques to maintain smooth interaction.

Common Patterns
---------------

**Adding Basic Objects:**

.. code-block:: python

   import viser
   
   server = viser.ViserServer()
   
   # Basic shapes
   server.scene.add_icosphere("sphere", radius=0.5, color=(255, 0, 0))
   server.scene.add_box("box", dimensions=(1, 1, 1), color=(0, 255, 0))
   server.scene.add_cylinder("cylinder", height=2.0, radius=0.3, color=(0, 0, 255))

**Positioning Objects:**

.. code-block:: python

   import viser.transforms as tf
   import numpy as np
   
   # Position and rotate objects
   server.scene.add_icosphere(
       "positioned_sphere",
       radius=0.3,
       position=(1.0, 2.0, 0.5),  # (x, y, z)
       wxyz=tf.SO3.from_z_radians(np.pi/4).wxyz  # 45° rotation around Z
   )

**Loading External Data:**

.. code-block:: python

   import trimesh
   
   # Load mesh from file
   mesh = trimesh.load_mesh("path/to/model.obj")
   server.scene.add_mesh_simple(
       "loaded_mesh",
       vertices=mesh.vertices,
       faces=mesh.faces
   )

Next Steps
----------

Once you're comfortable with scene fundamentals:

1. **Add Interactivity** → Browse ``examples_dev/02_gui/`` to build control panels
2. **Handle User Input** → Check ``examples_dev/03_interaction/`` to respond to clicks
3. **See Real Applications** → Explore ``examples_dev/04_demos/`` for complete examples

**Ready to start?** Run ``python examples_dev/01_scene/00_coordinate_frames.py`` to understand the basics!