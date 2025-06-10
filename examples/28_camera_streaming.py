"""Camera streaming

Demonstrates how to capture camera frames from the client, and send it to the server.
"""

import numpy as np

import viser


def main():
    CAPTURE_RES = (640, 480)

    server = viser.ViserServer()

    # Attach a camera stream handler to each client.
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # Camera configuration controls
        max_resolution_slider = client.gui.add_slider(
            "Max Resolution", min=240, max=2160, step=120, initial_value=720
        )
        fps_slider = client.gui.add_slider(
            "Frame Rate", min=1, max=60, step=1, initial_value=10
        )
        facing_mode_dropdown = client.gui.add_dropdown(
            "Camera", options=("user", "environment"), initial_value="user"
        )

        enable_cam_handle = client.gui.add_button("Enable Camera")

        client_id = client.client_id

        dummy_image = np.zeros((CAPTURE_RES[1], CAPTURE_RES[0], 3), dtype=np.uint8)
        client_image_handle = client.gui.add_image(dummy_image)
        server_image_handle = server.scene.add_image(
            name=f"/camera_frame_{client_id}",
            image=dummy_image,
            render_width=0.5,
            render_height=0.5,
            position=(client_id, 0, 0),
        )

        @client.gui.on_camera_stream_frame
        def _(event: viser.CameraStreamFrameEvent):
            client_image_handle.image = np.array(event.image)
            server_image_handle.image = np.array(event.image)

        @enable_cam_handle.on_click
        def _(_):
            client.configure_camera_stream(
                enabled=True,
                max_resolution=max_resolution_slider.value,
                frame_rate=fps_slider.value,
                facing_mode=facing_mode_dropdown.value,
            )
            enable_cam_handle.disabled = True

        def update_config():
            client.configure_camera_stream(
                enabled=True,
                max_resolution=max_resolution_slider.value,
                frame_rate=fps_slider.value,
                facing_mode=facing_mode_dropdown.value,
            )

        max_resolution_slider.on_update(lambda _: update_config())
        fps_slider.on_update(lambda _: update_config())
        facing_mode_dropdown.on_update(lambda _: update_config())

    server.sleep_forever()


if __name__ == "__main__":
    main()
