"""Camera streaming

Demonstrates how to capture camera frames from the client, and send it to the server.
"""

import io

import numpy as np
from PIL import Image

import viser


def main():
    CAPTURE_RES = (640, 480)

    server = viser.ViserServer()

    # Attach a camera stream handler to each client.
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        camera_enabled_handle = client.gui.add_checkbox(
            "Camera Enabled", initial_value=False
        )

        client_id = client.client_id

        dummy_image = np.zeros((CAPTURE_RES[1], CAPTURE_RES[0], 3), dtype=np.uint8)
        client_image_handle = client.gui.add_image(dummy_image)
        server_image_handle = server.scene.add_image(
            name=f"/camera_frame_{client_id}",
            image=dummy_image,
            render_width=100,
            render_height=CAPTURE_RES[1] / CAPTURE_RES[0] * 100,
            position=(0, 0, 0.5 * client_id),
        )

        client.configure_camera_stream(
            enabled=camera_enabled_handle.value,
            video_constraints={
                "width": CAPTURE_RES[0],
                "height": CAPTURE_RES[1],
                "facingMode": "user",
            },
            capture_fps=5.0,
            capture_resolution=CAPTURE_RES,
        )

        @client.on_camera_stream_frame
        def _(client: viser.ClientHandle, frame_message):
            image = Image.open(io.BytesIO(frame_message.frame_data))
            client_image_handle.image = np.array(image)
            server_image_handle.image = np.array(image)

        @camera_enabled_handle.on_update
        def _(_):
            enabled = camera_enabled_handle.value
            server.configure_camera_stream_all_clients(enabled=enabled)

    server.sleep_forever()


if __name__ == "__main__":
    main()
