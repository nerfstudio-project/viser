import React, { useEffect, useRef, useState, useContext, useCallback } from "react";
import { ViewerContext } from "./ViewerContext";
import { Button, Group, Popover } from "@mantine/core";
import Webcam from "react-webcam";

export function CameraStream() {
  const viewer = useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketConnected);
  const cameraEnabled = viewer.useGui((state) => state.cameraEnabled);
  const cameraReady = viewer.useGui((state) => state.cameraReady);
  const activeCameraRequest = viewer.useGui((state) => state.activeCameraRequest);
  const setCameraReady = viewer.useGui((state) => state.setCameraReady);
  const setCameraRequest = viewer.useGui((state) => state.setCameraRequest);
  const webcamRef = useRef<Webcam>(null);


  const captureFrameForRequest = useCallback((request: {
    request_id: string;
    max_resolution: number | null;
    facing_mode: "user" | "environment" | null;
    format: "image/jpeg" | "image/png";
  }) => {
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
  }, [viewer, cameraEnabled, cameraReady]);

  // Handle camera frame requests from GuiState
  useEffect(() => {
    if (activeCameraRequest) {
      console.log("📸 Received frame request:", activeCameraRequest.request_id);
      captureFrameForRequest(activeCameraRequest);
      // Clear the request after processing
      setCameraRequest(null);
    }
  }, [activeCameraRequest, captureFrameForRequest, setCameraRequest]);

  // Handle camera state changes
  useEffect(() => {
    if (cameraEnabled) {
      console.log("🎥 Enabling camera access");
      setCameraReady(false); // Reset ready state when enabling
    } else {
      console.log("🎥 Disabling camera access");
      setCameraReady(false);
    }
  }, [cameraEnabled, setCameraReady]);

  // Camera ready callback
  const handleUserMedia = useCallback(() => {
    console.log("🎥 Webcam stream started, waiting for stabilization...");
    // Add a small delay to ensure the camera is truly ready for capture
    setTimeout(() => {
      console.log("🎥 Webcam ready for on-demand capture");
      setCameraReady(true);
    }, 500); // 500ms delay to ensure camera is fully initialized
  }, [setCameraReady]);

  // Handle errors
  const handleUserMediaError = useCallback((error: any) => {
    console.error("❌ Webcam error:", error);
    setCameraReady(false);
  }, [setCameraReady]);

  // Reset camera ready state when disconnected
  useEffect(() => {
    if (!connected) {
      console.log("🔌 Server disconnected, camera no longer ready");
      setCameraReady(false);
    }
  }, [connected, setCameraReady]);

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
        videoConstraints={{
          facingMode: cameraEnabled ? "user" : "user",
          width: 1280,
          height: 720,
        }}
        onUserMedia={handleUserMedia}
        onUserMediaError={handleUserMediaError}
        mirrored={false}
        // This is a hack -- display: none doesn't work, it seems to fetch the current webcam render.
        style={{ position: "fixed", left: "-9999px", top: "-9999px" }}
      />
    </Group>
  );
}