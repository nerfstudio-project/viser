Embedding Visualizations
========================

Viser supports two main approaches for embedding 3D visualizations:

1. **Self-contained HTML** (Jupyter notebooks, myst-nb, static HTML files) - No server required
2. **Hosted playback** (GitHub Pages, custom hosting) - More flexible for large files

Self-Contained HTML (Jupyter & Static)
--------------------------------------

Viser provides a Plotly-like API for embedding 3D visualizations directly in
Jupyter notebooks or static HTML files. The embedded visualizations are
completely self-contained and work offline.

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

In Jupyter notebooks, you can display Viser visualizations directly:

.. code-block:: python

    import viser

    # Create server and add scene elements.
    server = viser.ViserServer()
    server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Option 1: Just return the server (uses _repr_html_).
    # This creates an iframe to the local server.
    server

    # Option 2: Use show() like Plotly's fig.show().
    server.show()

    # Option 3: Use embedded mode (works offline, like Plotly's include_plotlyjs).
    # This creates a self-contained HTML with all assets inlined.
    server.show(embed=True)

Static HTML Export
~~~~~~~~~~~~~~~~~~

For static HTML files (e.g., myst-nb documentation, standalone pages):

.. code-block:: python

    import viser
    from pathlib import Path

    server = viser.ViserServer()
    server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Generate self-contained HTML.
    html = server.get_embed_html(height=600, dark_mode=False)
    Path("visualization.html").write_text(html)

The resulting HTML file (~3-4MB) contains all JavaScript, CSS, and assets needed
to display the visualization, and works completely offline.

Animated Embeds
~~~~~~~~~~~~~~~

For animations, use :meth:`StateSerializer.insert_sleep` to add timing between
frames, then generate the embed from the serialized data:

.. code-block:: python

    import viser
    import numpy as np
    from pathlib import Path

    server = viser.ViserServer()
    box = server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Get the scene serializer for recording animations.
    serializer = server.get_scene_serializer()

    # Animate the box position over 3 seconds.
    for i in range(60):
        t = i / 60 * 2 * np.pi
        box.position = (np.sin(t), np.cos(t), 0.5)
        serializer.insert_sleep(3.0 / 60)  # 3 second total duration

    # Generate embed from serialized data.
    html = viser.get_embed_html_from_bytes(serializer.serialize())
    Path("animated.html").write_text(html)

The embedded HTML will include a playback timeline with play/pause controls
and a scrubber for navigating through the animation.

Embed Mode Limitations
~~~~~~~~~~~~~~~~~~~~~~

The self-contained embed mode has some limitations compared to the full Viser
client:

- **Gaussian splats**: Not supported (Web Workers cannot be inlined)
- **HDRIs**: All presets supported via CDN (requires internet connection)
- **Fonts**: Uses system fonts instead of Inter
- **GUI elements**: Scene-only mode (no interactive GUI panels)

These limitations keep the embed size manageable (~3-4MB) while supporting most
visualization use cases.


Hosted Playback (Advanced)
--------------------------

For larger visualizations or when you need full feature support, you can host
the Viser client separately and load scene data via URL.

.. warning::

   This workflow is more complex than self-contained embedding. We're
   documenting it nonetheless, since it's useful for advanced use cases.


Step 1: Exporting Scene State
-----------------------------

You can export static or dynamic 3D data from a Viser scene using the scene
serializer. :func:`viser.ViserServer.get_scene_serializer` returns a serializer
object that can serialize the current scene state to a binary format.

Static Scene Export
~~~~~~~~~~~~~~~~~~~

For static 3D visualizations, :func:`viser.infra.StateSerializer.serialize` can be used to
save the scene state:

.. code-block:: python

   import viser
   from pathlib import Path

   server = viser.ViserServer()

   # Add objects to the scene via server.scene
   # For example:
   # server.scene.add_mesh(...)
   # server.scene.add_point_cloud(...)
   server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

   # Serialize and save the scene state
   data = server.get_scene_serializer().serialize()  # Returns bytes
   Path("recording.viser").write_bytes(data)


As a suggestion, you can also add a button for exporting the scene state.
Clicking the button in your web browser will trigger a download of the
``.viser`` file.

.. code-block:: python

   import viser
   server = viser.ViserServer()

   # Add objects to the scene via server.scene.
   # For example:
   # server.scene.add_mesh(...)
   # server.scene.add_point_cloud(...)
   server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

   save_button = server.gui.add_button("Save Scene")

   @save_button.on_click
   def _(event: viser.GuiEvent) -> None:
       assert event.client is not None
       event.client.send_file_download("recording.viser", server.get_scene_serializer().serialize())

   server.sleep_forever()

Dynamic Scene Export
~~~~~~~~~~~~~~~~~~~~

For dynamic visualizations with animation, you can create a "3D video" by
calling :meth:`StateSerializer.insert_sleep` between frames:

.. code-block:: python

   import viser
   import numpy as np
   from pathlib import Path

   server = viser.ViserServer()

   # Add objects to the scene via server.scene
   # For example:
   # server.scene.add_mesh(...)
   # server.scene.add_point_cloud(...)
   box = server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

   # Create serializer.
   serializer = server.get_scene_serializer()

   num_frames = 100
   for t in range(num_frames):
       # Update existing scene objects or add new ones.
       box.position = (0.0, 0.0, np.sin(t / num_frames * 2 * np.pi))

       # Add a frame delay.
       serializer.insert_sleep(1.0 / 30.0)

   # Save the complete animation.
   data = serializer.serialize()  # Returns bytes
   Path("recording.viser").write_bytes(data)

.. note::
   Always add scene elements using :attr:`ViserServer.scene`, not :attr:`ClientHandle.scene`.

.. note::
   The ``.viser`` file is a binary format containing scene state data and is not meant to be human-readable.

Step 2: Creating a Viser Client Build
-------------------------------------

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


Step 4: Setting the initial camera pose
-----------------------------------------------

To set the initial camera pose, you can add a ``&logCamera`` parameter to the URL:

* ``http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/recordings/recording.viser&logCamera``

Then, open your Javascript console. You should see the camera pose printed
whenever you move the camera. It should look something like this:

* ``&initialCameraPosition=2.216,-4.233,-0.947&initialCameraLookAt=-0.115,0.346,-0.192&initialCameraUp=0.329,-0.904,0.272``

You can then add this string to the URL to set the initial camera pose.
