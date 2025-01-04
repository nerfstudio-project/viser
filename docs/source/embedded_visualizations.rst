Embedding Visualizations
===============================================

This guide describes how to export 3D visualizations from Viser and embed them into static webpages. The process involves three main steps: exporting scene state, creating a client build, and hosting the visualization.

.. warning::

   This workflow is experimental and not yet polished. We're documenting it
   nonetheless, since we think it's quite useful! If you have suggestions or
   improvements, issues and PRs are welcome.


Step 1: Exporting Scene State
----------------------------

You can export static or dynamic 3D data from a Viser scene using the scene
serializer. :func:`ViserServer.get_scene_serializer` returns a serializer
object that can serialize the current scene state to a binary format.

Static Scene Export
~~~~~~~~~~~~~~~~~~~

For static 3D visualizations, use the following code to save the scene state:

.. code-block:: python

   import viser
   from pathlib import Path

   server = viser.ViserServer()

   # Add objects to the scene via server.scene
   # For example:
   # server.scene.add_mesh(...)
   # server.scene.add_point_cloud(...)

   # Serialize and save the scene state
   data = server.get_scene_serializer().serialize()  # Returns bytes
   Path("recording.viser").write_bytes(data)


As a suggestion, you can also add a button for exporting the scene state:

.. code-block:: python

   server = viser.ViserServer()

   save_button.server.gui.add_button("Save Scene")

   @save_button.on_click
   def _(event: viser.GuiEvent) -> None:
       assert event.client is not None
       event.client.send_file_download("recording.viser", server.get_scene_serializer().serialize())

Dynamic Scene Export
~~~~~~~~~~~~~~~~~~~~

For dynamic visualizations with animation, you can create a "3D video" by inserting sleep commands between frames:

.. code-block:: python

   import viser
   from pathlib import Path

   server = server.ViserServer()

   # Add objects to the scene via server.scene
   # For example:
   # server.scene.add_mesh(...)
   # server.scene.add_point_cloud(...)

   serializer = server.get_scene_serializer()

   for t in range(num_frames):
       # Update existing scene objects or add new ones
       # server.scene.add_point_cloud(...)

       # Add a frame delay (e.g., for 30 FPS animation)
       serializer.insert_sleep(1 / 30)

   # Save the complete animation
   data = serializer.serialize()  # Returns bytes
   Path("recording.viser").write_bytes(data)

.. note::
   Always add scene elements using :attr:`ViserServer.scene`, not :attr:`ClientHandle.scene`.

.. note::
   The ``.viser`` file is a binary format containing scene state data and is not meant to be human-readable.

Step 2: Creating a Viser Client Build
-----------------------------------

To serve the 3D visualization, you'll need two things:

1. The ``.viser`` file containing your scene data
2. A build of the Viser client (static HTML/JS/CSS files)

With Viser installed, create the Viser client build using the command-line tool:

.. code-block:: bash

   # View available options
   viser-build-client --help

   # Build to a specific directory
   viser-build-client --output-dir viser-client/


Step 3: Hosting
---------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

For our hosting instructions, we're going to assume the following directory structure:

.. code-block::

    .
    ├── recordings/
    │   └── recording.viser    # Your exported scene data
    └── viser-client/
        ├── index.html         # Generated client files
        ├── assets/
        └── ...

This is just a suggestion; you can structure your files however you like.

Local Development Server
~~~~~~~~~~~~~~~~~~~~~~~~

For testing locally, you can use Python's built-in HTTP server:

.. code-block:: bash

    # Navigate to the parent directory containing both folders
    cd /path/to/parent/dir

    # Start the server (default port 8000)
    python -m http.server 8000

Then open your browser and navigate to:

* ``http://localhost:8000/viser-client/`` (default port)

This should show the a standard Viser client. To visualize the exported scene, you'll need to specify a URL via the ``?playbackPath=`` parameter:

* ``http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/recordings/recording.viser``


GitHub Pages Deployment
~~~~~~~~~~~~~~~~~~~~~~~

To host your visualization on GitHub Pages:

1. Create a new repository or use an existing one
2. Create a ``gh-pages`` branch or enable GitHub Pages on your main branch
3. Push your directory structure to the repository:

   .. code-block:: bash

       git add recordings/ viser-client/
       git commit -m "Add Viser visualization"
       git push origin main  # or gh-pages

Your visualization will be available at: ``https://user.github.io/repo/viser-client/?playbackPath=https://user.github.io/repo/recordings/recording.viser``

You can embed this into other webpages using an HTML ``<iframe />`` tag.


Step 4: (Extra) Setting the initial camera pose
-----------------------------------------------

To set the initial camera pose, you can add a ``&logCamera`` parameter to the URL:

* ``http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/recordings/recording.viser&logCamera``

Then, open your Javascript console. You should see the camera pose printed
whenever you move the camera. It should look something like this:

* ``&initialCameraPosition=2.216,-4.233,-0.947&initialCameraLookAt=-0.115,0.346,-0.192&initialCameraUp=0.329,-0.904,0.272``

You can then add this string to the URL to set the initial camera pose.
