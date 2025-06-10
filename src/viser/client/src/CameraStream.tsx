import { useEffect, useRef, useContext, useCallback } from "react";
import { ViewerContext } from "./ViewerContext";
import Webcam from "react-webcam";

export function CameraStream() {
  const viewer = useContext(ViewerContext)!;
  const viewerMutable = viewer.mutable.current; // Get mutable once.
  const connected = viewer.useGui((state) => state.websocketConnected);
  const cameraEnabled = viewer.useGui((state) => state.cameraEnabled);
  const cameraReady = viewer.useGui((state) => state.cameraReady);
  const activeCameraRequest = viewer.useGui((state) => state.activeCameraRequest);
  const setCameraReady = viewer.useGui((state) => state.setCameraReady);
  const setCameraRequest = viewer.useGui((state) => state.setCameraRequest);
  const webcamRef = useRef<Webcam>(null);

  // TODO: use the webcam props.

  // Handle camera frame capture requests.
  useEffect(() => {
    if (!activeCameraRequest) return;

    const request = activeCameraRequest;
    const timestamp = Date.now() / 1000;

    // Camera not enabled.
    if (!cameraEnabled) {
      viewerMutable.sendMessage({
        type: "CameraFrameResponseMessage",
        request_id: request.request_id,
        frame_data: null,
        timestamp: timestamp,
        error: "Camera access disabled",
        format: request.format,
      });
      setCameraRequest(null);
      return;
    }

    // Camera not found, or not ready.
    if (!webcamRef.current || !cameraReady) {
      viewerMutable.sendMessage({
        type: "CameraFrameResponseMessage",
        request_id: request.request_id,
        frame_data: null,
        timestamp: timestamp,
        error: "Camera not ready",
        format: request.format,
      });
      setCameraRequest(null);
      return;
    }

    const imageSrc = webcamRef.current.getScreenshot();

    // Tried to capture frame, but failed.
    if (!imageSrc) {
      viewerMutable.sendMessage({
        type: "CameraFrameResponseMessage",
        request_id: request.request_id,
        frame_data: null,
        timestamp: timestamp,
        error: "Failed to capture frame",
        format: request.format,
      });
      setCameraRequest(null);
      return;
    }

    // Convert base64 to Uint8Array.
    const byteString = atob(imageSrc.split(',')[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    const response = {
      type: "CameraFrameResponseMessage" as const,
      request_id: request.request_id,
      frame_data: ia,
      timestamp: timestamp,
      error: null,
      format: request.format || "image/jpeg" as const,
    };
    viewerMutable.sendMessage(response);

    // Clear the request after processing.
    console.log("🎥 Camera frame captured");
    setCameraRequest(null);
  }, [activeCameraRequest]);

  // Set camera to "ready" when enabled, by default.
  useEffect(() => {
    if (cameraEnabled) {
      setCameraReady(true);
    } else {
      setCameraReady(false);
    }
  }, [cameraEnabled]);

  // Let the error trigger the webcam to create a "enabled-but-not-ready" state.
  const handleUserMediaError = useCallback(() => { setCameraReady(false); }, []);

  // Reset camera ready state when disconnected.
  useEffect(() => {
    if (!connected) { setCameraReady(false); }
  }, [connected]);

  // Only render webcam if connected and enabled.
  if (!connected || !cameraEnabled) {
    return null;
  }

  return (
    <Webcam
      ref={webcamRef}
      audio={false}
      onUserMediaError={handleUserMediaError}
      mirrored={false}
      // This is a hack -- {display: none} doesn't work.
      // It seems to fetch the current webcam render.
      style={{ position: "fixed", left: "-9999px", top: "-9999px" }}
    />
  );
}