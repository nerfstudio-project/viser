Embedded Visualizations
========================

Viser supports embedding 3D visualizations in Jupyter notebooks, myst-nb
documentation, and standalone HTML files. Embedded visualizations are
self-contained and work offline.

Jupyter Notebooks & myst-nb
---------------------------

Use :meth:`~viser.ViserServer.show` to display visualizations inline:

.. code-block:: python

    import viser

    server = viser.ViserServer()
    server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Display inline (works in Jupyter notebooks and myst-nb docs).
    server.show()

    # Optional parameters:
    server.show(height=600, dark_mode=True)

The visualization is fully interactive with orbit controls, and works offline
once loaded.

Animated Visualizations
~~~~~~~~~~~~~~~~~~~~~~~

For animations, use :meth:`~viser.infra.StateSerializer.insert_sleep` to add
timing between frames:

.. code-block:: python

    import viser
    import numpy as np

    server = viser.ViserServer()
    box = server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Get the scene serializer for recording animations.
    serializer = server.get_scene_serializer()

    # Animate the box position over 3 seconds.
    for i in range(60):
        t = i / 60 * 2 * np.pi
        box.position = (np.sin(t), np.cos(t), 0.5)
        serializer.insert_sleep(3.0 / 60)  # 3 second total duration

    # Display the animation.
    serializer.show()

The embedded visualization includes a playback timeline with play/pause
controls and a scrubber for navigating through the animation.

Plain Python Scripts
--------------------

When running outside of IPython (e.g., a plain Python script),
:meth:`~viser.ViserServer.show` opens the visualization in your default web
browser:

.. code-block:: python

    import viser

    server = viser.ViserServer()
    server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Opens in default web browser.
    server.show()

Features & Limitations
----------------------

The embedded visualizations support most Viser scene features:

- **3D primitives**: Boxes, spheres, meshes, point clouds, etc.
- **Gaussian Splats**: Fully supported with WebAssembly sorting
- **HDRIs**: All 10 presets embedded (~480KB total)
- **Fonts**: Inter variable font for consistent typography
- **Animations**: Timeline with play/pause and scrubbing

**Limitations** compared to the full Viser client:

- **GUI elements**: Scene-only mode (no interactive GUI panels)
- **WebSocket connection**: Embeds are static snapshots, not live connections

Exporting .viser Files
----------------------

You can export scene data to ``.viser`` files for hosting or sharing:

.. code-block:: python

    import viser
    from pathlib import Path

    server = viser.ViserServer()
    server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    # Serialize and save the scene state.
    data = server.get_scene_serializer().serialize()
    Path("recording.viser").write_bytes(data)

You can also add a download button to export from the browser:

.. code-block:: python

    import viser

    server = viser.ViserServer()
    server.scene.add_box("/box", color=(255, 0, 0), dimensions=(1, 1, 1))

    save_button = server.gui.add_button("Save Scene")

    @save_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        data = server.get_scene_serializer().serialize()
        event.client.send_file_download("recording.viser", data)

    server.sleep_forever()

Hosted Playback (Advanced)
--------------------------

For hosting visualizations on a web server (e.g., GitHub Pages), you can serve
the Viser client with a ``.viser`` file:

1. **Build the client**:

   .. code-block:: bash

       viser-build-client --output-dir viser-client/

2. **Host with your .viser file**:

   .. code-block::

       .
       ├── recordings/
       │   └── recording.viser
       └── viser-client/
           ├── index.html
           └── ...

3. **Access via URL**:

   .. code-block::

       https://yoursite.com/viser-client/?playbackPath=../recordings/recording.viser

Camera Positioning
~~~~~~~~~~~~~~~~~~

To set the initial camera pose, add ``&logCamera`` to the URL to log camera
positions to the browser console as you navigate. Then add the logged
parameters to your URL:

.. code-block::

    ?playbackPath=...&initialCameraPosition=2.2,-4.2,-0.9&initialCameraLookAt=-0.1,0.3,-0.2&initialCameraUp=0.3,-0.9,0.3
