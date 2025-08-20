import { ViewerContext } from "./ViewerContext";
import { CameraControls, Instance, Instances } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import * as holdEvent from "hold-event";
import React, { useContext, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { PerspectiveCamera } from "three";
import * as THREE from "three";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { useThrottledMessageSender } from "./WebsocketUtils";
import { Grid, PivotControls } from "@react-three/drei";

function CrosshairVisual({
  visible,
  children,
}: {
  visible: boolean;
  children?: React.ReactNode;
}) {
  const { camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);

  const worldPos = new THREE.Vector3();
  useFrame(() => {
    if (groupRef.current && visible) {
      // Get world position of the crosshair.
      groupRef.current.getWorldPosition(worldPos);
      groupRef.current.scale.setScalar(
        camera.position.distanceTo(worldPos) / 20,
      );
    }
  });

  return (
    <group ref={groupRef} visible={visible}>
      <Instances limit={6}>
        <boxGeometry args={[0.4, 0.02, 0.02]} />
        <meshBasicMaterial color="#ff0000" opacity={0.3} transparent />
        {/* Horizontal line segments */}
        <Instance position={[0.5, 0.0, 0.0]} />
        <Instance position={[-0.5, 0.0, 0.0]} />
        <Instance
          position={[0.0, 0.0, 0.5]}
          rotation={new THREE.Euler(0.0, Math.PI / 2.0, 0.0)}
        />
        <Instance
          position={[0.0, 0.0, -0.5]}
          rotation={new THREE.Euler(0.0, Math.PI / 2.0, 0.0)}
        />
        {/* Vertical line segments */}
        <Instance
          position={[0.0, 0.5, 0.0]}
          rotation={new THREE.Euler(0.0, 0.0, Math.PI / 2.0)}
          color="#ff7700"
        />
        <Instance
          position={[0.0, -0.5, 0.0]}
          rotation={new THREE.Euler(0.0, 0.0, Math.PI / 2.0)}
          color="#ff7700"
        />
      </Instances>
      <mesh>
        <sphereGeometry args={[0.03, 8, 8]} />
        <meshBasicMaterial color="#ff0000" opacity={0.3} transparent />
      </mesh>
      {children}
    </group>
  );
}

function OrbitOriginTool({
  forceShow,
  pivotRef,
  onPivotChange,
  update,
}: {
  forceShow: boolean;
  pivotRef: React.RefObject<THREE.Group>;
  onPivotChange: (matrix: THREE.Matrix4) => void;
  update: () => void;
}) {
  const viewer = useContext(ViewerContext)!;
  const showOrbitOriginTool = viewer.useGui(
    (state) => state.showOrbitOriginTool,
  );
  const enableOrbitCrosshair = viewer.useDevSettings(
    (state) => state.enableOrbitCrosshair,
  );
  const showOrbitOriginCrosshair = viewer.useGui(
    (state) => state.showOrbitOriginCrosshair,
  );
  React.useEffect(update, [showOrbitOriginTool]);

  const show = showOrbitOriginTool || forceShow;
  return (
    <PivotControls
      ref={pivotRef}
      scale={200}
      lineWidth={3}
      fixed={true}
      axisColors={["#ffaaff", "#ff33ff", "#ffaaff"]}
      disableScaling={true}
      disableAxes={!show}
      disableRotations={!show}
      disableSliders={!show}
      onDragEnd={() => {
        onPivotChange(pivotRef.current!.matrix);
      }}
    >
      <Grid
        args={[10, 10, 10, 10]}
        infiniteGrid
        fadeStrength={0}
        fadeFrom={0}
        fadeDistance={1000}
        sectionColor={"#ffaaff"}
        cellColor={"#ffccff"}
        side={THREE.DoubleSide}
        visible={show}
      />
      {/* Crosshair visualization at look-at point */}
      <CrosshairVisual
        visible={enableOrbitCrosshair && showOrbitOriginCrosshair > 0}
      />
    </PivotControls>
  );
}

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);

  const sendCameraThrottled = useThrottledMessageSender(20).send;

  // Helper for resetting camera poses.
  const initialCameraRef = useRef<{
    camera: PerspectiveCamera;
    lookAt: THREE.Vector3;
  } | null>(null);

  const pivotRef = useRef<THREE.Group>(null);

  const viewerMutable = viewer.mutable.current;

  // Functions to handle crosshair visibility counter
  const showCrosshair = React.useCallback(() => {
    viewer.useGui.setState((state) => ({
      showOrbitOriginCrosshair: state.showOrbitOriginCrosshair + 1,
    }));
  }, [viewer]);

  const hideCrosshair = React.useCallback(() => {
    viewer.useGui.setState((state) => ({
      showOrbitOriginCrosshair: Math.max(0, state.showOrbitOriginCrosshair - 1),
    }));
  }, [viewer]);

  // Initialize to 0 on mount
  React.useEffect(() => {
    viewer.useGui.setState({ showOrbitOriginCrosshair: 0 });
  }, []);

  // Animation state interface.
  interface CameraAnimation {
    startUp: THREE.Vector3;
    targetUp: THREE.Vector3;
    startLookAt: THREE.Vector3;
    targetLookAt: THREE.Vector3;
    startTime: number;
    duration: number;
  }

  const [cameraAnimation, setCameraAnimation] =
    useState<CameraAnimation | null>(null);

  // Animation parameters.
  const ANIMATION_DURATION = 0.5; // seconds

  useFrame((state) => {
    if (cameraAnimation && viewerMutable.cameraControl) {
      const cameraControls = viewerMutable.cameraControl;
      const camera = cameraControls.camera;

      const elapsed = state.clock.getElapsedTime() - cameraAnimation.startTime;
      const progress = Math.min(elapsed / cameraAnimation.duration, 1);

      // Smooth step easing.
      const t = progress * progress * (3 - 2 * progress);

      // Interpolate up vector.
      const newUp = new THREE.Vector3()
        .copy(cameraAnimation.startUp)
        .lerp(cameraAnimation.targetUp, t)
        .normalize();

      // Interpolate look-at position.
      const newLookAt = new THREE.Vector3()
        .copy(cameraAnimation.startLookAt)
        .lerp(cameraAnimation.targetLookAt, t);

      camera.up.copy(newUp);

      // Back up position.
      const prevPosition = new THREE.Vector3();
      cameraControls.getPosition(prevPosition);

      cameraControls.updateCameraUp();

      // Restore position and set new look-at.
      cameraControls.setPosition(
        prevPosition.x,
        prevPosition.y,
        prevPosition.z,
        false,
      );

      cameraControls.setLookAt(
        prevPosition.x,
        prevPosition.y,
        prevPosition.z,
        newLookAt.x,
        newLookAt.y,
        newLookAt.z,
        false,
      );

      // Clear animation when complete.
      if (progress >= 1) {
        setCameraAnimation(null);
      }
    }
  });

  const { clock } = useThree();

  const updateCameraLookAtAndUpFromPivotControl = (matrix: THREE.Matrix4) => {
    if (!viewerMutable.cameraControl) return;

    const targetPosition = new THREE.Vector3();
    targetPosition.setFromMatrixPosition(matrix);

    const cameraControls = viewerMutable.cameraControl;
    const camera = viewerMutable.cameraControl.camera;

    // Get target up vector from matrix.
    const targetUp = new THREE.Vector3().setFromMatrixColumn(matrix, 1);

    // Get current look-at position.
    const currentLookAt = cameraControls.getTarget(new THREE.Vector3());

    // Start new animation.
    setCameraAnimation({
      startUp: camera.up.clone(),
      targetUp: targetUp,
      startLookAt: currentLookAt,
      targetLookAt: targetPosition,
      startTime: clock.getElapsedTime(),
      duration: ANIMATION_DURATION,
    });
  };

  const updatePivotControlFromCameraLookAtAndup = () => {
    if (cameraAnimation !== null) return;
    if (!viewerMutable.cameraControl) return;
    if (!pivotRef.current) return;

    const cameraControls = viewerMutable.cameraControl;
    const lookAt = cameraControls.getTarget(new THREE.Vector3());

    // Rotate matrix s.t. it's y-axis aligns with the camera's up vector.
    // We'll do this with math.
    const origRotation = new THREE.Matrix4().extractRotation(
      pivotRef.current.matrix,
    );

    const cameraUp = camera.up.clone().normalize();
    const pivotUp = new THREE.Vector3(0, 1, 0)
      .applyMatrix4(origRotation)
      .normalize();
    const axis = new THREE.Vector3()
      .crossVectors(pivotUp, cameraUp)
      .normalize();
    const angle = Math.acos(Math.min(1, Math.max(-1, cameraUp.dot(pivotUp))));

    // Create rotation matrix.
    const rotationMatrix = new THREE.Matrix4();
    if (axis.lengthSq() > 0.0001) {
      // Check if cross product is valid.
      rotationMatrix.makeRotationAxis(axis, angle);
    }
    // rotationMatrix.premultiply(origRotation);

    // Combine rotation with position.
    const matrix = new THREE.Matrix4();
    matrix.multiply(rotationMatrix);
    matrix.multiply(origRotation);
    matrix.setPosition(lookAt);

    pivotRef.current.matrix.copy(matrix);
    pivotRef.current.updateMatrixWorld(true);
  };

  viewerMutable.resetCameraView = () => {
    camera.up.set(
      initialCameraRef.current!.camera.up.x,
      initialCameraRef.current!.camera.up.y,
      initialCameraRef.current!.camera.up.z,
    );
    viewerMutable.cameraControl!.updateCameraUp();
    viewerMutable.cameraControl!.setLookAt(
      initialCameraRef.current!.camera.position.x,
      initialCameraRef.current!.camera.position.y,
      initialCameraRef.current!.camera.position.z,
      initialCameraRef.current!.lookAt.x,
      initialCameraRef.current!.lookAt.y,
      initialCameraRef.current!.lookAt.z,
      true,
    );
  };

  // Callback for sending cameras.
  // It makes the code more chaotic, but we preallocate a bunch of things to
  // minimize garbage collection!
  const R_threecam_cam = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI, 0.0, 0.0),
  );
  const R_world_threeworld = new THREE.Quaternion();
  const tmpMatrix4 = new THREE.Matrix4();
  const lookAt = new THREE.Vector3();
  const R_world_camera = new THREE.Quaternion();
  const t_world_camera = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const sendCamera = React.useCallback(() => {
    updatePivotControlFromCameraLookAtAndup();

    const three_camera = camera;
    const camera_control = viewerMutable.cameraControl;
    const canvas = viewerMutable.canvas!;

    if (camera_control === null) {
      // Camera controls not yet ready, let's re-try later.
      setTimeout(sendCamera, 10);
      return;
    }

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const T_world_threeworld = computeT_threeworld_world(viewer).invert();
    const T_world_camera = T_world_threeworld.clone()
      .multiply(
        tmpMatrix4
          .makeRotationFromQuaternion(three_camera.quaternion)
          .setPosition(three_camera.position),
      )
      .multiply(tmpMatrix4.makeRotationFromQuaternion(R_threecam_cam));
    R_world_threeworld.setFromRotationMatrix(T_world_threeworld);

    camera_control.getTarget(lookAt).applyQuaternion(R_world_threeworld);
    const up = three_camera.up.clone().applyQuaternion(R_world_threeworld);

    // Store initial camera values.
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        camera: three_camera.clone(),
        lookAt: camera_control.getTarget(new THREE.Vector3()),
      };
    }

    T_world_camera.decompose(t_world_camera, R_world_camera, scale);

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: t_world_camera.toArray(),
      image_height: canvas.height,
      image_width: canvas.width,
      fov: (three_camera.fov * Math.PI) / 180.0,
      near: three_camera.near,
      far: three_camera.far,
      look_at: [lookAt.x, lookAt.y, lookAt.z],
      up_direction: [up.x, up.y, up.z],
    });

    // Log camera.
    if (logCamera) {
      console.log(
        `&initialCameraPosition=${t_world_camera.x.toFixed(
          3,
        )},${t_world_camera.y.toFixed(3)},${t_world_camera.z.toFixed(3)}` +
          `&initialCameraLookAt=${lookAt.x.toFixed(3)},${lookAt.y.toFixed(
            3,
          )},${lookAt.z.toFixed(3)}` +
          `&initialCameraUp=${up.x.toFixed(3)},${up.y.toFixed(
            3,
          )},${up.z.toFixed(3)}`,
      );
    }
  }, [camera, sendCameraThrottled]);

  // Camera control search parameters.
  // EXPERIMENTAL: these may be removed or renamed in the future. Please pin to
  // a commit/version if you're relying on this (undocumented) feature.
  const searchParams = new URLSearchParams(window.location.search);
  const initialCameraPosString = searchParams.get("initialCameraPosition");
  const initialCameraLookAtString = searchParams.get("initialCameraLookAt");
  const initialCameraUpString = searchParams.get("initialCameraUp");
  const forceOrbitOriginTool = searchParams.get("forceOrbitOriginTool") === "1";
  const logCamera = viewer.useDevSettings((state) => state.logCamera);

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = viewer.useGui((state) => state.websocketConnected);
  const initialCameraPositionSet = React.useRef(false);
  React.useEffect(() => {
    if (!initialCameraPositionSet.current) {
      const initialCameraPos = new THREE.Vector3(
        ...((initialCameraPosString
          ? (initialCameraPosString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [3.0, 3.0, 3.0]) as [number, number, number]),
      );
      initialCameraPos.applyMatrix4(computeT_threeworld_world(viewer));
      const initialCameraLookAt = new THREE.Vector3(
        ...((initialCameraLookAtString
          ? (initialCameraLookAtString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0, 0, 0]) as [number, number, number]),
      );
      initialCameraLookAt.applyMatrix4(computeT_threeworld_world(viewer));
      const initialCameraUp = new THREE.Vector3(
        ...((initialCameraUpString
          ? (initialCameraUpString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0, 0, 1]) as [number, number, number]),
      );
      initialCameraUp.applyMatrix4(computeT_threeworld_world(viewer));
      initialCameraUp.normalize();

      camera.up.set(initialCameraUp.x, initialCameraUp.y, initialCameraUp.z);
      viewerMutable.cameraControl!.updateCameraUp();

      viewerMutable.cameraControl!.setLookAt(
        initialCameraPos.x,
        initialCameraPos.y,
        initialCameraPos.z,
        initialCameraLookAt.x,
        initialCameraLookAt.y,
        initialCameraLookAt.z,
        false,
      );
      initialCameraPositionSet.current = true;
    }

    viewerMutable.sendCamera = sendCamera;
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  // Send camera for 3D viewport changes.
  const canvas = viewerMutable.canvas!; // R3F canvas.
  React.useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);

    // Cleanup.
    return () => resizeObserver.disconnect();
  }, [canvas]);

  // Keyboard controls.
  React.useEffect(() => {
    const cameraControls = viewerMutable.cameraControl!;

    const keys = {
      w: new holdEvent.KeyboardKeyHold("KeyW", 1000 / 60),
      a: new holdEvent.KeyboardKeyHold("KeyA", 1000 / 60),
      s: new holdEvent.KeyboardKeyHold("KeyS", 1000 / 60),
      d: new holdEvent.KeyboardKeyHold("KeyD", 1000 / 60),
      q: new holdEvent.KeyboardKeyHold("KeyQ", 1000 / 60),
      e: new holdEvent.KeyboardKeyHold("KeyE", 1000 / 60),
      up: new holdEvent.KeyboardKeyHold("ArrowUp", 1000 / 60),
      down: new holdEvent.KeyboardKeyHold("ArrowDown", 1000 / 60),
      left: new holdEvent.KeyboardKeyHold("ArrowLeft", 1000 / 60),
      right: new holdEvent.KeyboardKeyHold("ArrowRight", 1000 / 60),
    };

    // TODO: these event listeners are currently never removed, even if this
    // component gets unmounted.
    keys.a.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.truck(-0.002 * event?.deltaTime, 0, false);
    });
    keys.d.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.truck(0.002 * event?.deltaTime, 0, false);
    });
    keys.w.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.forward(0.002 * event?.deltaTime, false);
    });
    keys.s.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.forward(-0.002 * event?.deltaTime, false);
    });
    keys.q.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.elevate(-0.002 * event?.deltaTime, false);
    });
    keys.e.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.elevate(0.002 * event?.deltaTime, false);
    });
    keys.left.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.rotate(
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    keys.right.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.rotate(
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    keys.up.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.rotate(
        0,
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });
    keys.down.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLDING, (event) => {
      cameraControls.rotate(
        0,
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });
    for (const key of Object.values(keys)) {
      key.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLD_START, () => {
        showCrosshair();
      });
      key.addEventListener(holdEvent.HOLD_EVENT_TYPE.HOLD_END, () => {
        hideCrosshair();
      });
    }

    // TODO: we currently don't remove any event listeners. This is a bit messy
    // because KeyboardKeyHold attaches listeners directly to the
    // document/window; it's unclear if we can remove these.
    return () => {
      return;
    };
  }, [CameraControls, showCrosshair, hideCrosshair]);

  return (
    <>
      <CameraControls
        ref={(controls) => (viewerMutable.cameraControl = controls)}
        minDistance={0.01}
        dollySpeed={0.3}
        smoothTime={0.05}
        draggingSmoothTime={0.0}
        onChange={sendCamera}
        onStart={() => {
          showCrosshair();
        }}
        onEnd={() => {
          hideCrosshair();
        }}
        makeDefault
      />
      <OrbitOriginTool
        forceShow={forceOrbitOriginTool}
        pivotRef={pivotRef}
        onPivotChange={(matrix) => {
          updateCameraLookAtAndUpFromPivotControl(matrix);
        }}
        update={updatePivotControlFromCameraLookAtAndup}
      />
    </>
  );
}
