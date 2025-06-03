Viser Server
============

The :class:`~viser.ViserServer` is the main entry point for creating interactive 3D visualizations. It manages the connection between your Python code and the web-based client.

Quick Start
-----------

.. code-block:: python

   import viser
   
   # Create a server (starts automatically on localhost:8080)
   server = viser.ViserServer()
   
   # Add a red sphere to the scene
   server.scene.add_icosphere(
       name="hello_sphere",
       radius=0.5,
       color=(255, 0, 0),
       position=(0, 0, 0)
   )
   
   # Add a GUI control
   with server.gui.add_folder("Controls"):
       size_slider = server.gui.add_slider("Size", min=0.1, max=2.0, step=0.1, initial_value=0.5)
   
   print("Visit http://localhost:8080 to see your visualization!")
   
   # Keep the server running
   while True:
       pass

Key Properties
--------------

The server provides access to the main APIs:

- :attr:`~viser.ViserServer.scene` - Add and manipulate 3D objects (:class:`~viser.SceneApi`)
- :attr:`~viser.ViserServer.gui` - Create user interface controls (:class:`~viser.GuiApi`)  
- :attr:`~viser.ViserServer.camera` - Control camera position and settings (:class:`~viser.CameraHandle`)

Common Patterns
---------------

**Basic Scene Setup:**

.. code-block:: python

   server = viser.ViserServer()
   
   # Add coordinate frame for reference
   server.scene.add_frame("world", axes_length=1.0, axes_radius=0.02)
   
   # Add some basic objects
   server.scene.add_icosphere("sphere", radius=0.3, color=(255, 0, 0), position=(1, 0, 0))
   server.scene.add_box("box", dimensions=(0.5, 0.5, 0.5), color=(0, 255, 0), position=(-1, 0, 0))

**Event Handling:**

.. code-block:: python

   @server.scene.on_click
   def handle_scene_click(event):
       print(f"Clicked on: {event.object_name}")
       print(f"Click position: {event.click_pos}")
   
   @server.on_client_connect
   def welcome_user(client):
       print(f"New client connected: {client.client_id}")

**Multi-Client Management:**

.. code-block:: python

   # Track connected clients
   connected_clients = set()
   
   @server.on_client_connect
   def handle_connect(client):
       connected_clients.add(client.client_id)
       print(f"Clients connected: {len(connected_clients)}")
   
   @server.on_client_disconnect
   def handle_disconnect(client):
       connected_clients.discard(client.client_id)
       print(f"Clients connected: {len(connected_clients)}")

Configuration Options
---------------------

**Port and Host:**

.. code-block:: python

   # Custom port
   server = viser.ViserServer(port=8081)
   
   # Custom host (for remote access)
   server = viser.ViserServer(host="0.0.0.0", port=8080)

**Verbose Logging:**

.. code-block:: python

   # Enable detailed logging
   server = viser.ViserServer(verbose=True)

API Reference
-------------

.. autoclass:: viser.ViserServer
   :members:
   :undoc-members:
