"""Camera on-demand capture

Demonstrates how to request camera frames from the client on-demand.
"""

import time

import numpy as np

import viser


def main():
    server = viser.ViserServer()

    # Attach camera capture handlers to each client.
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # Camera configuration controls
        max_resolution_slider = client.gui.add_slider(
            "Max Resolution", min=240, max=2160, step=120, initial_value=720
        )
        facing_mode_dropdown = client.gui.add_dropdown(
            "Camera", options=("user", "environment"), initial_value="user"
        )

        client_id = client.client_id

        # Create placeholder image displays
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        client_image_handle = client.gui.add_image(dummy_image)
        server.scene.add_transform_controls(
            name=f"/camera_frame_{client_id}",
            scale=0.2,
            position=(client_id, 0, 0),
            active_axes=(True, True, False),
        )
        server_image_handle = server.scene.add_image(
            name=f"/camera_frame_{client_id}/img",
            image=dummy_image,
            render_width=0.5,
            render_height=0.5,
            position=(0.25, 0.25, -0.001),
        )

        # "Race condition" seems fine when started from False?
        client.configure_camera_access(enabled=False)

        while True:
            try:
                image = client.capture_frame(
                    max_resolution=max_resolution_slider.value,
                    facing_mode=facing_mode_dropdown.value,
                    timeout=2.0,
                )
                client_image_handle.image = np.array(image)
                server_image_handle.image = np.array(image)
            except Exception as e:
                print(e)

            time.sleep(1 / 20)

    server.sleep_forever()


if __name__ == "__main__":
    main()
