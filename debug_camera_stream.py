#!/usr/bin/env python3

"""
Debug script for camera streaming
"""

import time
import viser

def main():
    server = viser.ViserServer()
    
    print("🚀 Starting debug server...")
    print("📍 Go to http://localhost:8080")
    
    # Simple controls
    enable_button = server.gui.add_button("Enable Camera Stream")
    disable_button = server.gui.add_button("Disable Camera Stream")
    status = server.gui.add_markdown("**Status:** Not started")
    
    frame_count = 0
    
    print("📋 Setting up camera stream callback...")
    
    @server.on_camera_stream_frame
    def handle_frame(client, frame_message):
        nonlocal frame_count
        frame_count += 1
        print(f"🎯 FRAME RECEIVED #{frame_count} from client {client.client_id}")
        print(f"    Size: {len(frame_message.frame_data)} bytes")
        print(f"    Dimensions: {frame_message.width}x{frame_message.height}")
        print(f"    Format: {frame_message.format}")
        print(f"    Timestamp: {frame_message.timestamp}")
        
        status.content = f"**Status:** Received {frame_count} frames"
    
    print("📋 Camera stream callback registered!")
    
    @enable_button.on_click
    def _(event):
        print("🔛 Enable button clicked")
        print(f"🔛 Current clients: {len(server.get_clients())}")
        
        server.configure_camera_stream_all_clients(
            enabled=True,
            video_constraints={
                "width": 640,
                "height": 480,
                "facingMode": "user"
            },
            capture_fps=5.0,  # Low FPS for debugging
            capture_resolution=(640, 480),
        )
        status.content = "**Status:** Camera stream enabled - waiting for frames..."
        print("🔛 Camera stream enabled!")
    
    @disable_button.on_click
    def _(event):
        print("🔴 Disable button clicked")
        server.configure_camera_stream_all_clients(enabled=False)
        status.content = "**Status:** Camera stream disabled"
        print("🔴 Camera stream disabled!")
    
    with server.gui.add_folder("Instructions"):
        server.gui.add_markdown("""
**Debug Instructions:**

1. Click "Enable Camera Stream"
2. Grant camera permissions when prompted
3. Check both:
   - Browser console (F12) for client-side logs
   - Python terminal for server-side logs
4. Look for:
   - "🎯 FRAME RECEIVED" messages in Python terminal
   - Camera permission prompts in browser
   - Video preview in top-right corner of browser

**Expected Flow:**
- Python: "📋 Camera stream callback registered!"
- Browser: Camera permission prompt
- Browser: Small video preview appears
- Browser console: "Sending frame to server" messages
- Python: "🎯 FRAME RECEIVED" messages
        """)
    
    print("✅ Debug server ready!")
    print("✅ Now open http://localhost:8080 and click 'Enable Camera Stream'")
    
    server.sleep_forever()

if __name__ == "__main__":
    main()