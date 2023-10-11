import AwaitLock from "await-lock";
import { CatmullRomLine, CubicBezierLine } from "@react-three/drei";
import { unpack } from "msgpackr";

import React, { useContext } from "react";
import * as THREE from "three";
import { TextureLoader } from "three";

import { ViewerContext } from "./App";
import { SceneNode } from "./SceneTree";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { CameraFrustum, CoordinateFrame, GlbAsset } from "./ThreeAssets";
import { Message } from "./WebsocketMessages";
import styled from "@emotion/styled";
import { Html, PivotControls } from "@react-three/drei";
import {
  isTexture,
  makeThrottledMessageSender,
  sendWebsocketMessage,
} from "./WebsocketFunctions";
import { isGuiConfig, useViserMantineTheme } from "./ControlPanel/GuiState";
import { useFrame } from "@react-three/fiber";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { MantineProvider, Paper } from "@mantine/core";

/** Convert raw RGB color buffers to linear color buffers. **/
function threeColorBufferFromUint8Buffer(colors: ArrayBuffer) {
  return new THREE.Float32BufferAttribute(
    new Float32Array(new Uint8Array(colors)).map((value) => {
      value = value / 255.0;
      if (value <= 0.04045) {
        return value / 12.92;
      } else {
        return Math.pow((value + 0.055) / 1.055, 2.4);
      }
    }),
    3,
  );
}

/** Returns a handler for all incoming messages. */
function useMessageHandler() {
  const viewer = useContext(ViewerContext)!;

  // TODO: we should clean this up.
  // https://github.com/nerfstudio-project/viser/issues/39
  const removeSceneNode = viewer.useSceneTree((state) => state.removeSceneNode);
  const resetScene = viewer.useSceneTree((state) => state.resetScene);
  const addSceneNode = viewer.useSceneTree((state) => state.addSceneNode);
  const setTheme = viewer.useGui((state) => state.setTheme);
  const addGui = viewer.useGui((state) => state.addGui);
  const addModal = viewer.useGui((state) => state.addModal);
  const removeModal = viewer.useGui((state) => state.removeModal);
  const removeGui = viewer.useGui((state) => state.removeGui);
  const setGuiValue = viewer.useGui((state) => state.setGuiValue);
  const setGuiVisible = viewer.useGui((state) => state.setGuiVisible);
  const setGuiDisabled = viewer.useGui((state) => state.setGuiDisabled);
  const setClickable = viewer.useSceneTree((state) => state.setClickable);

  // Same as addSceneNode, but make a parent in the form of a dummy coordinate
  // frame if it doesn't exist yet.
  function addSceneNodeMakeParents(node: SceneNode<any>) {
    const nodeFromName = viewer.useSceneTree.getState().nodeFromName;
    const parent_name = node.name.split("/").slice(0, -1).join("/");
    if (!(parent_name in nodeFromName)) {
      addSceneNodeMakeParents(
        new SceneNode<THREE.Group>(parent_name, (ref) => (
          <CoordinateFrame ref={ref} show_axes={false} />
        )),
      );
    }
    addSceneNode(node);
  }

  const mantineTheme = useViserMantineTheme();

  // Return message handler.
  return (message: Message) => {
    if (isGuiConfig(message)) {
      addGui(message);
      return;
    }

    switch (message.type) {
      // Request a render.
      case "GetRenderRequestMessage": {
        viewer.getRenderRequest.current = message;
        viewer.getRenderRequestState.current = "triggered";
        return;
      }
      // Configure the theme.
      case "ThemeConfigurationMessage": {
        setTheme(message);
        return;
      }

      // Enable/disable whether scene pointer events are sent.
      case "ScenePointerCallbackInfoMessage": {
        viewer.scenePointerCallbackCount.current += message.count;

        // Update cursor to indicate whether the scene can be clicked.
        viewer.canvasRef.current!.style.cursor =
          viewer.scenePointerCallbackCount.current > 0 ? "pointer" : "auto";
        return;
      }

      // Add a coordinate frame.
      case "FrameMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => (
            <CoordinateFrame
              ref={ref}
              show_axes={message.show_axes}
              axes_length={message.axes_length}
              axes_radius={message.axes_radius}
            />
          )),
        );
        return;
      }

      // Add a point cloud.
      case "PointCloudMessage": {
        const geometry = new THREE.BufferGeometry();
        const pointCloudMaterial = new THREE.PointsMaterial({
          size: message.point_size,
          vertexColors: true,
          toneMapped: false,
        });

        // Reinterpret cast: uint8 buffer => float32 for positions.
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.points.buffer.slice(
                message.points.byteOffset,
                message.points.byteOffset + message.points.byteLength,
              ),
            ),
            3,
          ),
        );
        geometry.computeBoundingSphere();

        // Wrap uint8 buffer for colors. Note that we need to set normalized=true.
        geometry.setAttribute(
          "color",
          threeColorBufferFromUint8Buffer(message.colors),
        );

        addSceneNodeMakeParents(
          new SceneNode<THREE.Points>(
            message.name,
            (ref) => (
              <points
                ref={ref}
                geometry={geometry}
                material={pointCloudMaterial}
              />
            ),
            () => {
              // TODO: we can switch to the react-three-fiber <bufferGeometry />,
              // <pointsMaterial />, etc components to avoid manual
              // disposal.
              geometry.dispose();
              pointCloudMaterial.dispose();
            },
          ),
        );
        return;
      }

      case "GuiModalMessage": {
        addModal(message);
        return;
      }

      case "GuiCloseModalMessage": {
        removeModal(message.id);
        return;
      }

      // Add mesh
      case "MeshMessage": {
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.MeshStandardMaterial({
          color: message.color || undefined,
          vertexColors: message.vertex_colors !== null,
          wireframe: message.wireframe,
          transparent: message.opacity !== null,
          opacity: message.opacity ?? undefined,
          side: {
            front: THREE.FrontSide,
            back: THREE.BackSide,
            double: THREE.DoubleSide,
          }[message.side],
        });
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.vertices.buffer.slice(
                message.vertices.byteOffset,
                message.vertices.byteOffset + message.vertices.byteLength,
              ),
            ),
            3,
          ),
        );
        if (message.vertex_colors !== null) {
          geometry.setAttribute(
            "color",
            threeColorBufferFromUint8Buffer(message.vertex_colors),
          );
        }

        geometry.setIndex(
          new THREE.Uint32BufferAttribute(
            new Uint32Array(
              message.faces.buffer.slice(
                message.faces.byteOffset,
                message.faces.byteOffset + message.faces.byteLength,
              ),
            ),
            1,
          ),
        );
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        addSceneNodeMakeParents(
          new SceneNode<THREE.Mesh>(
            message.name,
            (ref) => {
              return <mesh ref={ref} geometry={geometry} material={material} />;
            },
            () => {
              // TODO: we can switch to the react-three-fiber <bufferGeometry />,
              // <meshStandardMaterial />, etc components to avoid manual
              // disposal.
              geometry.dispose();
              material.dispose();
            },
          ),
        );
        return;
      }
      // Add a camera frustum.
      case "CameraFrustumMessage": {
        const texture =
          message.image_media_type !== null &&
          message.image_base64_data !== null
            ? new TextureLoader().load(
                `data:${message.image_media_type};base64,${message.image_base64_data}`,
              )
            : undefined;

        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => (
              <CameraFrustum
                ref={ref}
                fov={message.fov}
                aspect={message.aspect}
                scale={message.scale}
                color={message.color}
                image={texture}
              />
            ),
            () => texture?.dispose(),
          ),
        );
        return;
      }
      case "TransformControlsMessage": {
        const name = message.name;
        const sendDragMessage = makeThrottledMessageSender(
          viewer.websocketRef,
          50,
        );
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => (
            <group onClick={(e) => e.stopPropagation()}>
              <PivotControls
                ref={ref}
                scale={message.scale}
                lineWidth={message.line_width}
                fixed={message.fixed}
                autoTransform={message.auto_transform}
                activeAxes={message.active_axes}
                disableAxes={message.disable_axes}
                disableSliders={message.disable_sliders}
                disableRotations={message.disable_rotations}
                translationLimits={message.translation_limits}
                rotationLimits={message.rotation_limits}
                depthTest={message.depth_test}
                opacity={message.opacity}
                onDrag={(l) => {
                  const attrs = viewer.nodeAttributesFromName.current;
                  if (attrs[message.name] === undefined) {
                    attrs[message.name] = {};
                  }

                  const wxyz = new THREE.Quaternion();
                  wxyz.setFromRotationMatrix(l);
                  const position = new THREE.Vector3().setFromMatrixPosition(l);

                  const nodeAttributes = attrs[message.name]!;
                  nodeAttributes.wxyz = [wxyz.w, wxyz.x, wxyz.y, wxyz.z];
                  nodeAttributes.position = position.toArray();
                  sendDragMessage({
                    type: "TransformControlsUpdateMessage",
                    name: name,
                    wxyz: nodeAttributes.wxyz,
                    position: nodeAttributes.position,
                  });
                }}
              />
            </group>
          )),
        );
        return;
      }
      case "SetCameraLookAtMessage": {
        const cameraControls = viewer.cameraControlRef.current!;

        const R_threeworld_world = new THREE.Quaternion();
        R_threeworld_world.setFromEuler(
          new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0),
        );
        const target = new THREE.Vector3(
          message.look_at[0],
          message.look_at[1],
          message.look_at[2],
        );
        target.applyQuaternion(R_threeworld_world);
        cameraControls.setTarget(target.x, target.y, target.z, false);
        return;
      }
      case "SetCameraUpDirectionMessage": {
        const camera = viewer.cameraRef.current!;
        const cameraControls = viewer.cameraControlRef.current!;
        const R_threeworld_world = new THREE.Quaternion();
        R_threeworld_world.setFromEuler(
          new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0),
        );
        const updir = new THREE.Vector3(
          message.position[0],
          message.position[1],
          message.position[2],
        ).applyQuaternion(R_threeworld_world);
        camera.up.set(updir.x, updir.y, updir.z);

        // Back up position.
        const prevPosition = new THREE.Vector3();
        cameraControls.getPosition(prevPosition);

        cameraControls.updateCameraUp();

        // Restore position, which gets unexpectedly mutated in updateCameraUp().
        cameraControls.setPosition(
          prevPosition.x,
          prevPosition.y,
          prevPosition.z,
          false,
        );
        return;
      }
      case "SetCameraPositionMessage": {
        const cameraControls = viewer.cameraControlRef.current!;

        // Set the camera position. Note that this will shift the orientation as-well.
        const position_cmd = new THREE.Vector3(
          message.position[0],
          message.position[1],
          message.position[2],
        );
        const R_worldthree_world = new THREE.Quaternion();
        R_worldthree_world.setFromEuler(
          new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0),
        );
        position_cmd.applyQuaternion(R_worldthree_world);

        cameraControls.setPosition(
          position_cmd.x,
          position_cmd.y,
          position_cmd.z,
        );
        return;
      }
      case "SetCameraFovMessage": {
        const camera = viewer.cameraRef.current!;
        // tan(fov / 2.0) = 0.5 * film height / focal length
        // focal length = 0.5 * film height / tan(fov / 2.0)
        camera.setFocalLength(
          (0.5 * camera.getFilmHeight()) / Math.tan(message.fov / 2.0),
        );
        return;
      }
      case "SetOrientationMessage": {
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[message.name] === undefined) attr[message.name] = {};
        attr[message.name]!.wxyz = message.wxyz;
        break;
      }
      case "SetPositionMessage": {
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[message.name] === undefined) attr[message.name] = {};
        attr[message.name]!.position = message.position;
        break;
      }
      case "SetSceneNodeVisibilityMessage": {
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[message.name] === undefined) attr[message.name] = {};
        attr[message.name]!.visibility = message.visible;
        break;
      }
      // Add a background image.
      case "BackgroundImageMessage": {
        new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_rgb}`,
          (texture) => {
            texture.encoding = THREE.sRGBEncoding;

            const oldBackgroundTexture =
              viewer.backgroundMaterialRef.current!.uniforms.colorMap.value;
            viewer.backgroundMaterialRef.current!.uniforms.colorMap.value =
              texture;
            if (isTexture(oldBackgroundTexture)) oldBackgroundTexture.dispose();

            viewer.useGui.setState({ backgroundAvailable: true });
          },
        );
        viewer.backgroundMaterialRef.current!.uniforms.enabled.value = true;
        viewer.backgroundMaterialRef.current!.uniforms.hasDepth.value =
          message.base64_depth !== null;

        if (message.base64_depth !== null) {
          // If depth is available set the texture
          new TextureLoader().load(
            `data:image/png;base64,${message.base64_depth}`,
            (texture) => {
              const oldDepthTexture =
                viewer.backgroundMaterialRef.current?.uniforms.depthMap.value;
              viewer.backgroundMaterialRef.current!.uniforms.depthMap.value =
                texture;
              if (isTexture(oldDepthTexture)) oldDepthTexture.dispose();
            },
          );
        }
        return;
      }
      // Add a 2D label.
      case "LabelMessage": {
        const Label = styled.span`
          background-color: rgba(255, 255, 255, 0.85);
          padding: 0.2em;
          border-radius: 0.2em;
          border: 1px solid #777;
          color: #333;

          &:before {
            content: "";
            position: absolute;
            top: -1em;
            left: 1em;
            width: 0;
            height: 0;
            border-left: 1px solid #777;
            box-sizing: border-box;
            height: 0.8em;
            box-shadow: 0 0 1em 0.1em rgba(255, 255, 255, 1);
          }
        `;
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => {
              // We wrap with <group /> because Html doesn't implement THREE.Object3D.
              return (
                <group ref={ref}>
                  <Html>
                    <div
                      style={{
                        width: "10em",
                        fontSize: "0.8em",
                        transform: "translateX(-1em) translateY(1em)",
                      }}
                    >
                      <Label>{message.text}</Label>
                    </div>
                  </Html>
                </group>
              );
            },
            undefined,
            true,
          ),
        );
        return;
      }
      case "Gui3DMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => {
              // We wrap with <group /> because Html doesn't implement THREE.Object3D.
              return (
                <group ref={ref}>
                  <Html prepend={false}>
                    <MantineProvider
                      withGlobalStyles
                      withNormalizeCSS
                      theme={mantineTheme}
                    >
                      <Paper
                        sx={{
                          width: "20em",
                          fontSize: "0.8em",
                          marginLeft: "1em",
                          marginTop: "1em",
                        }}
                        shadow="md"
                        onPointerDown={(evt) => {
                          evt.stopPropagation();
                        }}
                      >
                        <GeneratedGuiContainer
                          containerId={message.container_id}
                          viewer={viewer}
                        />
                      </Paper>
                    </MantineProvider>
                  </Html>
                </group>
              );
            },
            undefined,
            true,
          ),
        );
        return;
      }
      // Add an image.
      case "ImageMessage": {
        // It's important that we load the texture outside of the node
        // construction callback; this prevents flickering by ensuring that the
        // texture is ready before the scene tree updates.
        new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_data}`,
          (texture) => {
            // TODO: this onLoad callback prevents flickering, but could cause messages to be handled slightly out-of-order.
            addSceneNodeMakeParents(
              new SceneNode<THREE.Group>(
                message.name,
                (ref) => {
                  return (
                    <group ref={ref}>
                      <mesh rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}>
                        <planeGeometry
                          attach="geometry"
                          args={[message.render_width, message.render_height]}
                        />
                        <meshBasicMaterial
                          attach="material"
                          transparent={true}
                          side={THREE.DoubleSide}
                          map={texture}
                          toneMapped={false}
                        />
                      </mesh>
                    </group>
                  );
                },
                () => texture.dispose(),
              ),
            );
          },
        );
        return;
      }
      // Remove a scene node by name.
      case "RemoveSceneNodeMessage": {
        console.log("Removing scene node:", message.name);
        removeSceneNode(message.name);
        return;
      }
      // Set the clickability of a particular scene node.
      case "SetSceneNodeClickableMessage": {
        // This setTimeout is totally unnecessary, but can help surface some race
        // conditions.
        setTimeout(() => setClickable(message.name, message.clickable), 50);
        return;
      }
      // Reset the entire scene, removing all scene nodes.
      case "ResetSceneMessage": {
        resetScene();

        const oldBackground = viewer.sceneRef.current?.background;
        viewer.sceneRef.current!.background = null;
        if (isTexture(oldBackground)) oldBackground.dispose();

        viewer.useGui.setState({ backgroundAvailable: false });
        // Disable the depth texture rendering
        viewer.backgroundMaterialRef.current!.uniforms.enabled.value = false;
        return;
      }
      // Set the value of a GUI input.
      case "GuiSetValueMessage": {
        setGuiValue(message.id, message.value);
        return;
      }
      // Set the hidden state of a GUI input.
      case "GuiSetVisibleMessage": {
        setGuiVisible(message.id, message.visible);
        return;
      }
      // Set the disabled state of a GUI input.
      case "GuiSetDisabledMessage": {
        setGuiDisabled(message.id, message.disabled);
        return;
      }
      // Remove a GUI input.
      case "GuiRemoveMessage": {
        removeGui(message.id);
        return;
      }
      // Add a glTF/GLB asset.
      case "GlbMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <GlbAsset
                ref={ref}
                glb_data={new Uint8Array(message.glb_data)}
                scale={message.scale}
              />
            );
          }),
        );
        return;
      }
      case "CatmullRomSplineMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <group ref={ref}>
                <CatmullRomLine
                  points={message.positions}
                  closed={message.closed}
                  curveType={message.curve_type}
                  tension={message.tension}
                  lineWidth={message.line_width}
                  color={message.color}
                ></CatmullRomLine>
              </group>
            );
          }),
        );
        return;
      }
      case "CubicBezierSplineMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <group ref={ref}>
                {[...Array(message.positions.length - 1).keys()].map((i) => (
                  <CubicBezierLine
                    key={i}
                    start={message.positions[i]}
                    end={message.positions[i + 1]}
                    midA={message.control_points[2 * i]}
                    midB={message.control_points[2 * i + 1]}
                    lineWidth={message.line_width}
                    color={message.color}
                  ></CubicBezierLine>
                ))}
              </group>
            );
          }),
        );
        return;
      }
      default: {
        console.log("Received message did not match any known types:", message);
        return;
      }
    }
  };
}

export function FrameSynchronizedMessageHandler() {
  const handleMessage = useMessageHandler();
  const viewer = useContext(ViewerContext)!;
  const messageQueueRef = viewer.messageQueueRef;

  // We'll reuse the same canvas.
  const renderBufferCanvas = React.useMemo(() => new OffscreenCanvas(1, 1), []);

  useFrame(() => {
    // Send a render along if it was requested!
    if (viewer.getRenderRequestState.current === "triggered") {
      viewer.getRenderRequestState.current = "pause";
    } else if (viewer.getRenderRequestState.current === "pause") {
      const sourceCanvas = viewer.canvasRef.current!;

      const targetWidth = viewer.getRenderRequest.current!.width;
      const targetHeight = viewer.getRenderRequest.current!.height;

      // We'll save a render to an intermediate canvas with the requested dimensions.
      if (renderBufferCanvas.width !== targetWidth)
        renderBufferCanvas.width = targetWidth;
      if (renderBufferCanvas.height !== targetHeight)
        renderBufferCanvas.height = targetHeight;

      const ctx = renderBufferCanvas.getContext("2d")!;
      ctx.reset();
      // Use a white background for JPEGs, which don't have an alpha channel.
      if (viewer.getRenderRequest.current?.format === "image/jpeg") {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, renderBufferCanvas.width, renderBufferCanvas.height);
      }

      // Determine offsets for the source canvas. We'll always center our renders.
      // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
      let sourceWidth = sourceCanvas.width;
      let sourceHeight = sourceCanvas.height;

      const sourceAspect = sourceWidth / sourceHeight;
      const targetAspect = targetWidth / targetHeight;

      if (sourceAspect > targetAspect) {
        // The source is wider than the target.
        // We need to shrink the width.
        sourceWidth = Math.round(targetAspect * sourceHeight);
      } else if (sourceAspect < targetAspect) {
        // The source is narrower than the target.
        // We need to shrink the height.
        sourceHeight = Math.round(sourceWidth / targetAspect);
      }

      console.log(
        `Sending render; requested aspect ratio was ${targetAspect} (dimensinos: ${targetWidth}/${targetHeight}), copying from aspect ratio ${
          sourceWidth / sourceHeight
        } (dimensions: ${sourceWidth}/${sourceHeight}).`,
      );

      ctx.drawImage(
        sourceCanvas,
        (sourceCanvas.width - sourceWidth) / 2.0,
        (sourceCanvas.height - sourceHeight) / 2.0,
        sourceWidth,
        sourceHeight,
        0,
        0,
        targetWidth,
        targetHeight,
      );

      viewer.getRenderRequestState.current = "in_progress";

      // Encode the image, the send it.
      renderBufferCanvas
        .convertToBlob({
          type: viewer.getRenderRequest.current!.format,
          quality: viewer.getRenderRequest.current!.quality / 100.0,
        })
        .then(async (blob) => {
          if (blob === null) {
            console.error("Render failed");
            viewer.getRenderRequestState.current = "ready";
            return;
          }
          const payload = new Uint8Array(await blob.arrayBuffer());
          sendWebsocketMessage(viewer.websocketRef, {
            type: "GetRenderResponseMessage",
            payload: payload,
          });
          viewer.getRenderRequestState.current = "ready";
        });
    }

    // Handle messages, but only if we're not trying to render something.
    if (viewer.getRenderRequestState.current === "ready") {
      // Handle messages before every frame.
      // Place this directly in ws.onmessage can cause race conditions!
      //
      // If a render is requested, note that we don't handle any more messages
      // until the render is done.
      const requestRenderIndex = messageQueueRef.current.findIndex(
        (message) => message.type === "GetRenderRequestMessage",
      );
      const numMessages =
        requestRenderIndex !== -1
          ? requestRenderIndex + 1
          : messageQueueRef.current.length;
      const processBatch = messageQueueRef.current.splice(0, numMessages);
      processBatch.forEach(handleMessage);
    }
  });

  return null;
}

/** Component for handling websocket connections. */
export function WebsocketMessageProducer() {
  const messageQueueRef = useContext(ViewerContext)!.messageQueueRef;
  const viewer = useContext(ViewerContext)!;
  const server = viewer.useGui((state) => state.server);
  const resetGui = viewer.useGui((state) => state.resetGui);

  syncSearchParamServer(server);

  React.useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let ws: null | WebSocket = null;
    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws = new WebSocket(server);

      // Timeout is necessary when we're connecting to an SSH/tunneled port.
      const retryTimeout = setTimeout(() => {
        ws?.close();
      }, 5000);

      ws.onopen = () => {
        clearTimeout(retryTimeout);
        console.log(`Connected!${server}`);
        viewer.websocketRef.current = ws;
        viewer.useGui.setState({ websocketConnected: true });
      };

      ws.onclose = (event) => {
        console.log(`Disconnected! ${server} code=${event.code}`);
        clearTimeout(retryTimeout);
        viewer.websocketRef.current = null;
        viewer.scenePointerCallbackCount.current = 0;
        viewer.useGui.setState({ websocketConnected: false });
        resetGui();

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      ws.onmessage = async (event) => {
        // Reduce websocket backpressure.
        const messagePromise = new Promise<Message[]>((resolve) => {
          (event.data.arrayBuffer() as Promise<ArrayBuffer>).then((buffer) => {
            resolve(unpack(new Uint8Array(buffer)) as Message[]);
          });
        });

        // Try our best to handle messages in order. If this takes more than 1 second, we give up. :)
        await orderLock.acquireAsync({ timeout: 1000 }).catch(() => {
          console.log("Order lock timed out.");
          orderLock.release();
        });
        try {
          const messages = await messagePromise;
          messageQueueRef.current.push(...messages);
        } finally {
          orderLock.acquired && orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      viewer.useGui.setState({ websocketConnected: false });
      ws?.close();
      clearTimeout(timeout);
    };
  }, [server, resetGui]);

  return <></>;
}
