import React, { useEffect, useRef, useState, useContext } from "react";
import { ViewerContext } from "./ViewerContext";

export function CameraStream() {
  console.log("🎥 CameraStream component rendering...");
  
  const viewer = useContext(ViewerContext)!;
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use state for config so component re-renders when it changes
  const [config, setConfig] = useState(viewer.mutable.current.cameraStreamConfig);
  
  console.log("🎥 Initial config:", config);

  // Watch for config changes in the mutable ref - but only check enabled state
  useEffect(() => {
    const checkConfigChanges = () => {
      const currentConfig = viewer.mutable.current.cameraStreamConfig;
      
      // Only update if enabled state actually changed (ignore other properties)
      if (currentConfig.enabled !== config.enabled) {
        console.log("📝 Config enabled changed, updating state:", currentConfig.enabled);
        setConfig({...currentConfig}); // Create new object to trigger re-render
      }
    };
    
    // Use a longer interval to reduce rapid refresh in Safari
    const interval = setInterval(checkConfigChanges, 2000); // Check every 2s
    return () => clearInterval(interval);
  }, [config.enabled]); // Only depend on enabled state

  const startCapture = async () => {
    try {
      console.log("🎥 startCapture called");
      console.log("🎥 Current location:", window.location);
      console.log("🎥 Protocol:", window.location.protocol);
      console.log("🎥 Hostname:", window.location.hostname);
      console.log("🎥 User agent:", navigator.userAgent);
      console.log("🎥 Browser info:", {
        isChrome: navigator.userAgent.includes('Chrome'),
        isSafari: navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome'),
        isFirefox: navigator.userAgent.includes('Firefox'),
        isLocalhost: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      });
      
      // Check if mediaDevices is available
      if (!navigator.mediaDevices) {
        throw new Error("navigator.mediaDevices not available - requires HTTPS or localhost");
      }
      
      if (!navigator.mediaDevices.getUserMedia) {
        throw new Error("getUserMedia not supported by this browser");
      }

      console.log("🎥 Media devices available:", {
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!navigator.mediaDevices.getUserMedia,
        enumerateDevices: !!navigator.mediaDevices.enumerateDevices
      });

      // Check permission state first
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'camera' as PermissionName });
        console.log("🎥 Camera permission state:", permissionStatus.state);
        
        if (permissionStatus.state === 'denied') {
          const isChrome = navigator.userAgent.includes('Chrome');
          const errorMessage = isChrome 
            ? "Camera permission denied. To fix this in Chrome:\n" +
              "1. Click the camera icon in the address bar\n" +
              "2. Select 'Allow' for camera access\n" +
              "3. Refresh the page and try again\n" +
              "Or go to Settings > Privacy > Camera and allow this site"
            : "Camera permission denied. Please allow camera access in browser settings.";
          throw new Error(errorMessage);
        }
      } catch (e) {
        console.warn("🎥 Could not check permission state:", e);
        // If permission query fails, still try to access camera - some browsers don't support it
      }

      // Check available devices
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        console.log("🎥 Available video devices:", videoDevices.length);
        console.log("🎥 Video devices:", videoDevices);
        
        if (videoDevices.length === 0) {
          throw new Error("No camera devices found");
        }
      } catch (e) {
        console.warn("🎥 Could not enumerate devices:", e);
      }

      // For HTTP localhost, we need to be more explicit about constraints
      const isLocalhost = window.location.hostname === 'localhost' || 
                         window.location.hostname === '127.0.0.1';
      
      const videoConstraints = config.videoConstraints || {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      };

      console.log("🎥 Requesting camera access with constraints:", videoConstraints);
      console.log("🎥 Is localhost:", isLocalhost);
      console.log("🎥 HTTPS:", window.location.protocol === 'https:');

      // Try with different constraint approaches
      let stream: MediaStream | null = null;
      
      // First try with the specified constraints
      try {
        console.log("🎥 Attempt 1: Using specified constraints");
        stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false,
        });
        console.log("🎥 Success with specified constraints");
      } catch (e1) {
        console.warn("🎥 Failed with specified constraints:", e1);
        
        // Try with simpler constraints
        try {
          console.log("🎥 Attempt 2: Using simple constraints");
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
          console.log("🎥 Success with simple constraints");
        } catch (e2) {
          console.warn("🎥 Failed with simple constraints:", e2);
          
          // Try with basic width/height only
          try {
            console.log("🎥 Attempt 3: Using basic constraints");
            stream = await navigator.mediaDevices.getUserMedia({
              video: { width: 640, height: 480 },
              audio: false,
            });
            console.log("🎥 Success with basic constraints");
          } catch (e3) {
            console.error("🎥 All camera access attempts failed");
            throw e3;
          }
        }
      }
      
      if (!stream) {
        throw new Error("Failed to obtain camera stream");
      }
      
      console.log("🎥 Camera stream obtained successfully:", stream);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        setError(null);

        // Start frame capture
        const captureInterval = setInterval(() => {
          captureFrame();
        }, 1000 / (config.captureFps || 10));
        intervalRef.current = captureInterval;
      }
    } catch (err) {
      let errorMessage = "Camera access failed";
      
      if (err instanceof Error) {
        const isChrome = navigator.userAgent.includes('Chrome');
        const isDeniedError = err.name === 'NotAllowedError' || err.message.includes('denied') || err.message.includes('permission');
        
        if (isDeniedError && isChrome) {
          errorMessage = "Camera permission denied. To fix this in Chrome:\n" +
            "1. Click the camera icon in the address bar\n" +
            "2. Select 'Allow' for camera access\n" +
            "3. Refresh the page and try again\n" +
            "Or go to Settings > Privacy > Camera and allow this site";
        } else if (isDeniedError) {
          errorMessage = "Camera permission denied. Please allow camera access in your browser settings and refresh the page.";
        } else {
          errorMessage = err.message;
        }
        
        console.error("Camera error details:", {
          name: err.name,
          message: err.message,
          hostname: window.location.hostname,
          protocol: window.location.protocol,
          isChrome,
          isDeniedError
        });
      }
      
      setError(errorMessage);
      console.error("Error accessing camera:", err);
    }
  };

  const stopCapture = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsStreaming(false);
  };

  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      console.log("Cannot capture frame - video not ready:", {
        video: !!video,
        canvas: !!canvas,
        videoWidth: video?.videoWidth,
        videoHeight: video?.videoHeight
      });
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.error("Cannot get canvas context");
      return;
    }

    // Set canvas dimensions
    const width = config.captureResolution ? config.captureResolution[0] : video.videoWidth;
    const height = config.captureResolution ? config.captureResolution[1] : video.videoHeight;
    
    canvas.width = width;
    canvas.height = height;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, width, height);

    // Convert to blob and send to server
    canvas.toBlob((blob) => {
      if (blob) {
        console.log(`Capturing frame: ${width}x${height}, ${blob.size} bytes`);
        const reader = new FileReader();
        reader.onload = () => {
          const arrayBuffer = reader.result as ArrayBuffer;
          const uint8Array = new Uint8Array(arrayBuffer);
          
          console.log(`Sending frame to server: ${uint8Array.length} bytes`);
          
          const message = {
            type: "CameraStreamFrameMessage" as const,
            frame_data: uint8Array,
            timestamp: Date.now() / 1000,
            width,
            height,
            format: "image/jpeg" as const,
          };
          
          console.log("Message object:", message);
          console.log("SendMessage function:", viewer.mutable.current.sendMessage);
          
          try {
            viewer.mutable.current.sendMessage(message);
            console.log("Message sent successfully");
          } catch (error) {
            console.error("Error sending message:", error);
          }
        };
        reader.readAsArrayBuffer(blob);
      } else {
        console.error("Failed to create blob from canvas");
      }
    }, "image/jpeg", 0.8);
  };

  useEffect(() => {
    console.log("🎬 Config enabled changed:", config.enabled, "isStreaming:", isStreaming);
    
    if (config.enabled && !isStreaming) {
      console.log("🎬 Starting camera capture...");
      // Add small delay to prevent rapid start/stop cycles in Safari
      const timeoutId = setTimeout(() => {
        startCapture();
      }, 100);
      return () => clearTimeout(timeoutId);
    } else if (!config.enabled && isStreaming) {
      console.log("🎬 Stopping camera capture...");
      stopCapture();
    }

    return () => {
      stopCapture();
    };
  }, [config.enabled, isStreaming]);

  useEffect(() => {
    return () => {
      stopCapture();
    };
  }, []);

  if (!config.enabled) {
    return null;
  }

  return (
    <div style={{ position: "fixed", top: "10px", right: "10px", zIndex: 1000, background: "rgba(0,0,0,0.8)", padding: "10px", borderRadius: "5px" }}>
      {error && (
        <div style={{ 
          color: "red", 
          fontSize: "12px", 
          padding: "4px", 
          marginBottom: "5px",
          whiteSpace: "pre-line",
          backgroundColor: "rgba(255, 255, 255, 0.9)",
          borderRadius: "3px",
          maxWidth: "300px"
        }}>
          Camera Error: {error}
        </div>
      )}
      {config.enabled && (
        <div style={{ color: "white", fontSize: "10px", marginBottom: "5px" }}>
          Camera Stream: {isStreaming ? "Active" : "Connecting..."}
        </div>
      )}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{ 
          display: config.enabled ? "block" : "none",
          width: "160px",
          height: "120px",
          border: "1px solid #ccc"
        }}
        onLoadedMetadata={() => {
          if (videoRef.current) {
            videoRef.current.play();
            console.log("Video loaded and playing");
          }
        }}
        onPlaying={() => {
          console.log("Video is playing");
        }}
        onError={(e) => {
          console.error("Video error:", e);
        }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}