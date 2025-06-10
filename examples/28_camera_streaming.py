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

        capture_button = client.gui.add_button("📸 Capture Frame")
        camera_access_checkbox = client.gui.add_checkbox(
            "Enable Camera Access", initial_value=False
        )
        continuous_capture_checkbox = client.gui.add_checkbox(
            "Continuous Capture (20 FPS)", initial_value=False
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

        # State for continuous capture
        continuous_capture_active = False

        @capture_button.on_click
        def _(_):
            print(f"📸 Requesting frame from client {client_id}")
            capture_button.disabled = True
            capture_button.name = "Capturing..."

            try:
                # Request frame on-demand
                image = client.capture_frame(
                    max_resolution=max_resolution_slider.value,
                    facing_mode=facing_mode_dropdown.value,
                    timeout=10.0,
                )
                print(f"✅ Received {image.size} frame from client {client_id}")

                # Update displays
                client_image_handle.image = np.array(image)
                server_image_handle.image = np.array(image)

            except Exception as e:
                print(f"❌ Failed to capture frame: {e}")
            finally:
                capture_button.disabled = False
                capture_button.name = "📸 Capture Frame"

        @camera_access_checkbox.on_update
        def _(_):
            enabled = camera_access_checkbox.value
            print(
                f"📷 {'Enabling' if enabled else 'Disabling'} camera access for client {client_id}"
            )
            client.configure_camera_access(enabled=enabled)

        # while True:
        #     if continuous_capture_checkbox.value:
        #         frame = client.capture_frame(
        #             max_resolution=max_resolution_slider.value,
        #             facing_mode=facing_mode_dropdown.value,
        #         )
        #         client_image_handle.image = np.array(frame)
        #         server_image_handle.image = np.array(frame)
        #         time.sleep(0.05)

        @continuous_capture_checkbox.on_update
        def _(_):
            nonlocal continuous_capture_active

            if continuous_capture_checkbox.value:
                # Start continuous capture burst
                print(f"🎬 Starting continuous capture burst for client {client_id}")
                continuous_capture_active = True

                # Disable single capture button during continuous mode
                capture_button.disabled = True
                capture_button.name = "📸 (Continuous Mode Active)"

                # Capture frames in a simple loop (20 FPS for 5 seconds = 100 frames)
                for i in range(100):
                    if not continuous_capture_checkbox.value:  # Check if still enabled
                        break

                    try:
                        # Request frame with short timeout for continuous mode
                        image = client.capture_frame(
                            max_resolution=max_resolution_slider.value,
                            facing_mode=facing_mode_dropdown.value,
                            timeout=2.0,
                        )

                        # Update displays
                        client_image_handle.image = np.array(image)
                        server_image_handle.image = np.array(image)

                        print(f"📸 Continuous frame {i+1}/100")

                    except Exception as e:
                        print(f"❌ Continuous capture error: {e}")

                    # Wait for next frame (20 FPS = 50ms interval)
                    time.sleep(0.05)

                # Re-enable single capture button when done
                continuous_capture_active = False
                capture_button.disabled = False
                capture_button.name = "📸 Capture Frame"
                continuous_capture_checkbox.value = False  # Auto-uncheck when done
                print(f"✅ Continuous capture burst completed for client {client_id}")
            else:
                # Stop continuous capture (will break the loop)
                print(f"⏹️ Stopping continuous capture for client {client_id}")
                continuous_capture_active = False

    server.sleep_forever()


if __name__ == "__main__":
    main()
