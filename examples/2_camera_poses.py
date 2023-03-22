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
            def camera_update(camera: viser.CameraState) -> None:
                print("New camera", camera)

            # Show the client ID in the GUI.
            client.add_gui_text("Info", initial_value=f"Client {id}", disabled=True)

        # We can also just read the camera state with client.get_camera().
        camera = client.get_camera()
        print(f"Camera pose for client {id}")
        print(f"\twxyz: {camera.wxyz}")
        print(f"\tposition: {camera.position}")
        print(f"\tfov: {camera.fov}")
        print(f"\taspect: {camera.aspect}")
        print(f"\tlast update: {camera.last_updated}")

    time.sleep(1.0)
