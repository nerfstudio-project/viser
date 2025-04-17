export type ViewerContextContents = {
  messageSource: "websocket" | "file_playback";
  // Zustand hooks.
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  // Useful references.
  sendMessageRef: React.MutableRefObject<(message: Message) => void>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  backgroundMaterialRef: React.MutableRefObject<THREE.ShaderMaterial | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
  sendCameraRef: React.MutableRefObject<(() => void) | null>;
  resetCameraViewRef: React.MutableRefObject<(() => void) | null>; // Ensure this allows null
  // Scene node attributes.
  nodeAttributesFromName: React.MutableRefObject<{
// ... rest of interface

} 