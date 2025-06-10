import React, { useEffect, useRef, useState, useContext, useCallback } from "react";
import { ViewerContext } from "./ViewerContext";
import { Button, Group, Popover } from "@mantine/core";
import Webcam from "react-webcam";

export function CameraStream() {
  const viewer = useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketConnected);
  const webcamRef = useRef<Webcam>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);


  const captureFrameForRequest = useCallback((request: any) => {
    if (!cameraEnabled) {
      console.log("🎥 Camera access disabled");
      // Send error response
      viewer.mutable.current.sendMessage({
        type: "CameraFrameResponseMessage",
        request_id: request.request_id,
        frame_data: null,
        timestamp: Date.now() / 1000,
        error: "Camera access disabled",
        format: request.format || "image/jpeg",
      });
      return;
    }

    if (!webcamRef.current || !cameraReady) {
      console.log("🎥 Camera not ready for capture");
      // Send error response
      viewer.mutable.current.sendMessage({
        type: "CameraFrameResponseMessage",
        request_id: request.request_id,
        frame_data: null,
        timestamp: Date.now() / 1000,
        error: "Camera not ready",
        format: request.format || "image/jpeg",
      });
      return;
    }

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      console.log("🎥 Failed to capture frame");
      // Send error response
      viewer.mutable.current.sendMessage({
        type: "CameraFrameResponseMessage",
        request_id: request.request_id,
        frame_data: null,
        timestamp: Date.now() / 1000,
        error: "Failed to capture frame",
        format: request.format || "image/jpeg",
      });
      return;
    }

    console.log("📸 Captured frame for request:", request.request_id);

    // Convert base64 to Uint8Array
    const byteString = atob(imageSrc.split(',')[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    // Send successful response
    const response = {
      type: "CameraFrameResponseMessage" as const,
      request_id: request.request_id,
      frame_data: ia,
      timestamp: Date.now() / 1000,
      error: null,
      format: request.format || "image/jpeg" as const,
    };

    viewer.mutable.current.sendMessage(response);
    console.log("✅ Frame response sent");
  }, [viewer, cameraReady]);

  // Handle camera frame requests
  useEffect(() => {
    const handleFrameRequest = (event: CustomEvent) => {
      const message = event.detail;
      console.log("📸 Received frame request:", message.request_id);
      captureFrameForRequest(message);
    };

    window.addEventListener('cameraFrameRequest', handleFrameRequest as EventListener);
    
    return () => {
      window.removeEventListener('cameraFrameRequest', handleFrameRequest as EventListener);
    };
  }, [captureFrameForRequest]);

  // Handle camera enable/disable events
  useEffect(() => {
    const handleEnableCamera = () => {
      console.log("🎥 Enabling camera access");
      setCameraEnabled(true);
    };
    
    const handleDisableCamera = () => {
      console.log("🎥 Disabling camera access");
      setCameraEnabled(false);
    };

    window.addEventListener('enableCamera', handleEnableCamera);
    window.addEventListener('disableCamera', handleDisableCamera);
    
    return () => {
      window.removeEventListener('enableCamera', handleEnableCamera);
      window.removeEventListener('disableCamera', handleDisableCamera);
    };
  }, []);

  // Camera ready callback
  const handleUserMedia = useCallback(() => {
    console.log("🎥 Webcam ready for on-demand capture");
    setCameraReady(true);
    window.dispatchEvent(new CustomEvent('cameraReady'));
  }, []);

  // Handle errors
  const handleUserMediaError = useCallback((error: any) => {
    console.error("❌ Webcam error:", error);
    setCameraReady(false);
    window.dispatchEvent(new CustomEvent('cameraError'));
  }, []);

  // Reset camera ready state when disconnected or disabled
  useEffect(() => {
    if (!connected) {
      console.log("🔌 Server disconnected, camera no longer ready");
      setCameraReady(false);
    }
  }, [connected]);

  useEffect(() => {
    if (!cameraEnabled) {
      console.log("🎥 Camera disabled, resetting ready state");
      setCameraReady(false);
      window.dispatchEvent(new CustomEvent('cameraDisabled'));
    }
  }, [cameraEnabled]);

  // Only render webcam if connected and enabled
  if (!connected || !cameraEnabled) {
    return null;
  }

  return (
    <Group justify="center">
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: "user" }} // Default constraints
        onUserMedia={handleUserMedia}
        onUserMediaError={handleUserMediaError}
        mirrored={false}
        // This is a hack -- display: none doesn't work, it seems to fetch the current webcam render.
        style={{ position: "fixed", left: "-9999px", top: "-9999px" }}
      />
    </Group>
  );
}