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
        enable_cam_handle = client.gui.add_button("Enable Camera")

        client_id = client.client_id

        dummy_image = np.zeros((CAPTURE_RES[1], CAPTURE_RES[0], 3), dtype=np.uint8)
        client_image_handle = client.gui.add_image(dummy_image)
        server_image_handle = server.scene.add_image(
            name=f"/camera_frame_{client_id}",
            image=dummy_image,
            render_width=CAPTURE_RES[0] / CAPTURE_RES[1] * 0.5,
            render_height=0.5,
            position=(client_id, 0, 0),
        )

        @client.gui.on_camera_stream_frame
        def _(event: viser.CameraStreamFrameEvent):
            client_image_handle.image = np.array(event.image)
            server_image_handle.image = np.array(event.image)

        @enable_cam_handle.on_click
        def _(_):
            client.configure_camera_stream()
            enable_cam_handle.disabled = True

    server.sleep_forever()


if __name__ == "__main__":
    main()
