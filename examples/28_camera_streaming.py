"""Camera streaming

Demonstrates how to use camera streaming to capture frames from client webcams or phone cameras.
Frames are sent from the client to the server where they can be processed in real-time.
"""

import time
import numpy as np
import io
from PIL import Image

import viser

def main() -> None:
    server = viser.ViserServer()

    # GUI controls for camera streaming
    with server.gui.add_folder("Camera Stream Controls"):
        enable_stream = server.gui.add_checkbox("Enable Camera Stream", initial_value=False)
        capture_fps = server.gui.add_slider("Capture FPS", min=1, max=30, step=1, initial_value=10)
        capture_width = server.gui.add_number("Capture Width", initial_value=640, min=320, max=1920)
        capture_height = server.gui.add_number("Capture Height", initial_value=480, min=240, max=1080)
        
    # Status display
    with server.gui.add_folder("Stream Status", expand_by_default=True):
        status_text = server.gui.add_markdown("**Status:** Camera stream disabled")
        frames_received = server.gui.add_markdown("**Frames received:** 0")
        last_frame_info = server.gui.add_markdown("**Last frame:** None")
        debug_info = server.gui.add_markdown("""
- Open browser dev tools (F12) to see console logs
- Camera requires permission on first use
- For HTTP: only localhost/127.0.0.1 allows camera access
- Small video preview will appear in top-right when active
        """)

    # Statistics
    frame_count = 0
    last_update_time = time.time()

    def update_camera_config():
        """Update camera streaming configuration for all clients."""
        enabled = enable_stream.value
        resolution = (int(capture_width.value), int(capture_height.value)) if enabled else None
        
        print(f"🔧 Updating camera config: enabled={enabled}, resolution={resolution}")
        
        server.configure_camera_stream_all_clients(
            enabled=enabled,
            video_constraints={
                "width": int(capture_width.value),
                "height": int(capture_height.value),
                "facingMode": "user"  # Use front-facing camera by default
            } if enabled else None,
            capture_fps=float(capture_fps.value) if enabled else None,
            capture_resolution=resolution,
        )
        
        print(f"🔧 Camera config sent to {len(server.get_clients())} clients")
        
        # Update status
        if enabled:
            status_text.content = f"**Status:** Camera stream enabled ({capture_width.value}x{capture_height.value} @ {capture_fps.value}fps)"
        else:
            status_text.content = "**Status:** Camera stream disabled"

    # Callback handlers for GUI controls
    @enable_stream.on_update
    def _(_):
        update_camera_config()

    @capture_fps.on_update
    def _(_):
        if enable_stream.value:
            update_camera_config()

    @capture_width.on_update
    def _(_):
        if enable_stream.value:
            update_camera_config()

    @capture_height.on_update  
    def _(_):
        if enable_stream.value:
            update_camera_config()

    # Camera stream frame handler
    print("📋 Registering server-level camera stream callback...")
    
    @server.on_camera_stream_frame
    def handle_camera_frame(client: viser.ClientHandle, frame_message):
        nonlocal frame_count, last_update_time
        frame_count += 1
        
        print(f"🎯 CALLBACK TRIGGERED! Received camera frame {frame_count} from client {client.client_id}: {len(frame_message.frame_data)} bytes")
        
        # Decode the frame data
        try:
            # Frame data is JPEG encoded
            image = Image.open(io.BytesIO(frame_message.frame_data))
            width, height = image.size
            
            print(f"Frame {frame_count}: {width}x{height} pixels")
            
            # Update statistics every 10 frames or every 2 seconds
            current_time = time.time()
            if frame_count % 10 == 0 or (current_time - last_update_time) > 2.0:
                frames_received.content = f"**Frames received:** {frame_count}"
                last_frame_info.content = f"**Last frame:** {width}x{height}, {len(frame_message.frame_data)} bytes, from client {client.client_id}"
                last_update_time = current_time
                
            # Process the frame here
            # Example: Convert to numpy array for processing
            frame_array = np.array(image)
            
            # Example processing: Display the frame as a scene image
            # (Note: This will update rapidly and may cause performance issues)
            if frame_count % 5 == 0:  # Only update every 5th frame to avoid performance issues
                # Convert PIL image back to bytes for viser
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG', quality=80)
                img_bytes.seek(0)
                
                # Display in the 3D scene
                server.scene.add_image(
                    f"/camera_feed_client_{client.client_id}",
                    img_bytes.read(),
                    render_width=2.0,
                    render_height=2.0 * height / width,
                    position=(0.0, 0.0, 1.0),
                )
                print(f"Updated scene with frame {frame_count}")
                
        except Exception as e:
            print(f"Error processing camera frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()

    # Information text
    with server.gui.add_folder("Instructions", expand_by_default=True):
        server.gui.add_markdown(
            """
            **Camera Streaming Example**

            This example demonstrates how to use camera streaming to capture frames from client webcams or phone cameras.
            Frames are sent from the client to the server where they can be processed in real-time.

            **Instructions:**
            
            1. **Enable camera streaming** by checking the "Enable Camera Stream" checkbox
            2. **Grant camera permissions** when prompted by your browser
            3. **Adjust settings** like FPS and resolution as needed
            4. **View the live feed** as it appears in the 3D scene
            
            **Features:**
            - Real-time camera frame capture from client devices
            - Configurable frame rate and resolution  
            - Frame processing on the server side
            - Display of received frames in the 3D scene
            
            **Use Cases:**
            - Remote monitoring and surveillance
            - Computer vision applications
            - Real-time image analysis
            - Interactive camera-based experiences
            
            **Note:** Camera streaming requires HTTPS in production environments. 
            For development, localhost connections allow camera access over HTTP.
            """,
        )

    # Keep the server running
    server.sleep_forever()

if __name__ == "__main__":
    main()