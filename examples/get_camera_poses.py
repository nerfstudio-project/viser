import time

import viser

server = viser.ViserServer()

while True:
    # Get all currently connected clients.
    client_ids = server.get_client_ids()
    print("Connected client IDs", client_ids)

    for client_id in client_ids:
        # Try to print camera poses from each connected client.
        client_info = server.get_client_info(client_id)
        if client_info is not None:
            print(f"Camera pose for client {client_id}")
            print(f"\twxyz: {client_info.camera.wxyz}")
            print(f"\tposition: {client_info.camera.position}")
            print(f"\tlast update: {client_info.camera_timestamp}")

    time.sleep(1.0)
