"""Example reading camera poses from connected clients."""

import time

import viser

server = viser.ViserServer()

seen_clients = set()
while True:
    # Get all currently connected clients.
    clients = server.get_clients()
    print("Connected client IDs", clients.keys())

    for id, client in clients.items():
        # New client? We can attach a callback.
        if id not in seen_clients:
            seen_clients.add(id)

            # This will run whenever we get a new camera!
            @client.on_camera_update
            def camera_update(client: viser.ClientHandle) -> None:
                print("New camera", client.camera)

            # Show the client ID in the GUI.
            gui_info = client.add_gui_text("Info", initial_value=f"Client {id}")
            gui_info.disabled = True

        camera = client.camera
        print(f"Camera pose for client {id}")
        print(f"\twxyz: {camera.wxyz}")
        print(f"\tposition: {camera.position}")
        print(f"\tfov: {camera.fov}")
        print(f"\taspect: {camera.aspect}")
        print(f"\tlast update: {camera.update_timestamp}")

    time.sleep(1.0)
