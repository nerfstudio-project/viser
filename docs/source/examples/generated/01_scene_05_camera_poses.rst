Camera poses
============

Example showing how we can detect new clients and read camera poses from them.

**Source:** ``examples_dev/01_scene/05_camera_poses.py``

.. figure:: ../_static/examples/01_scene_05_camera_poses.png
   :width: 100%
   :alt: Camera poses

Code
----

.. code-block:: python
   :linenos:

   """Camera poses
   
   Example showing how we can detect new clients and read camera poses from them.
   """
   
   import time
   
   import viser
   
   server = viser.ViserServer()
   server.scene.world_axes.visible = True
   
   
   @server.on_client_connect
   def _(client: viser.ClientHandle) -> None:
       print("new client!")
   
       # This will run whenever we get a new camera!
       @client.camera.on_update
       def _(_: viser.CameraHandle) -> None:
           print(f"New camera on client {client.client_id}!")
   
       # Show the client ID in the GUI.
       gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
       gui_info.disabled = True
   
   
   while True:
       # Get all currently connected clients.
       clients = server.get_clients()
       print("Connected client IDs", clients.keys())
   
       for id, client in clients.items():
           print(f"Camera pose for client {id}")
           print(f"\twxyz: {client.camera.wxyz}")
           print(f"\tposition: {client.camera.position}")
           print(f"\tfov: {client.camera.fov}")
           print(f"\taspect: {client.camera.aspect}")
           print(f"\tlast update: {client.camera.update_timestamp}")
           print(
               f"\tcanvas size: {client.camera.image_width}x{client.camera.image_height}"
           )
   
       time.sleep(2.0)
   
