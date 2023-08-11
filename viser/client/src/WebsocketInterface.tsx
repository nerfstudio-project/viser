import AwaitLock from "await-lock";
import { unpack } from "msgpackr";

import React, { useContext } from "react";
import * as THREE from "three";
import { TextureLoader } from "three";

import { ViewerContext } from "./App";
import { SceneNode } from "./SceneTree";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { CameraFrustum, CoordinateFrame } from "./ThreeAssets";
import { Message } from "./WebsocketMessages";
import styled from "@emotion/styled";
import { Html, PivotControls } from "@react-three/drei";
import { isTexture, makeThrottledMessageSender } from "./WebsocketFunctions";
import { isGuiConfig } from "./ControlPanel/GuiState";
import { useFrame } from "@react-three/fiber";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Paper } from "@mantine/core";
/** Float **/
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
  const removeGui = viewer.useGui((state) => state.removeGui);
  const removeGuiContainer = viewer.useGui((state) => state.removeGuiContainer);
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

  // Return message handler.
  return (message: Message) => {
    if (isGuiConfig(message)) {
      addGui(message);
      return;
    }

    switch (message.type) {
      case "ThemeConfigurationMessage": {
        setTheme(message);
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

        const height = message.scale * Math.tan(message.fov / 2.0) * 2.0;

        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => (
              <group ref={ref}>
                <CameraFrustum
                  fov={message.fov}
                  aspect={message.aspect}
                  scale={message.scale}
                  color={message.color}
                />
                {texture && (
                  <mesh
                    position={[0.0, 0.0, message.scale]}
                    rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
                  >
                    <planeGeometry
                      attach="geometry"
                      args={[message.aspect * height, height]}
                    />
                    <meshBasicMaterial
                      attach="material"
                      transparent={true}
                      side={THREE.DoubleSide}
                      map={texture}
                    />
                  </mesh>
                )}
              </group>
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
        cameraControls.setTarget(target.x, target.y, target.z);
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
        console.log("Background");
        new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_data}`,
          (texture) => {
            // TODO: this onLoad callback prevents flickering, but could cause messages to be handled slightly out-of-order.
            texture.encoding = THREE.sRGBEncoding;

            const oldBackground = viewer.sceneRef.current?.background;
            viewer.sceneRef.current!.background = texture;
            if (isTexture(oldBackground)) oldBackground.dispose();

            viewer.useGui.setState({ backgroundAvailable: true });
          },
        );
        return;
      }
      // Add a camera-aligned RGBD image
      case "PopupImageMessage": {
        viewer.nerfMaterialRef.current!.uniforms.enabled.value = true;
        new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_rgb}`,
          (texture) => {
            // TODO: this onLoad callback prevents flickering, but could cause messages to be handled slightly out-of-order.
            texture.encoding = THREE.sRGBEncoding;
            viewer.nerfMaterialRef.current!.uniforms.nerfColor.value = texture;
          }
        );
        new TextureLoader().load(
          `data:$image/png;base64,${message.base64_depth}`,
          (texture) => {
            // TODO: this onLoad callback prevents flickering, but could cause messages to be handled slightly out-of-order.);
            // texture.format = THREE.RGBAIntegerFormat;
            // texture.type = THREE.UnsignedByteType;
            // texture.internalFormat = 'RGBA8UI';
            texture.minFilter = THREE.NearestFilter;
            texture.magFilter = THREE.NearestFilter;
            viewer.nerfMaterialRef.current!.uniforms.nerfDepth.value = texture;
            viewer.nerfMaterialRef.current!.uniforms.depthScale.value = message.depth_scale;
          }
        );
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
          new SceneNode<THREE.Group>(message.name, (ref) => {
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
          }),
        );
        return;
      }
      case "Gui3DMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            // We wrap with <group /> because Html doesn't implement THREE.Object3D.
            return (
              <group ref={ref}>
                <Html>
                  <Paper
                    sx={{
                      width: "20em",
                      fontSize: "0.8em",
                    }}
                    withBorder
                  >
                    <GeneratedGuiContainer
                      containerId={message.container_id}
                      viewer={viewer}
                    />
                  </Paper>
                </Html>
              </group>
            );
          }),
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
        setClickable(message.name, message.clickable);
        return;
      }
      // Reset the entire scene, removing all scene nodes.
      case "ResetSceneMessage": {
        resetScene();

        const oldBackground = viewer.sceneRef.current?.background;
        viewer.sceneRef.current!.background = null;
        if (isTexture(oldBackground)) oldBackground.dispose();

        viewer.useGui.setState({ backgroundAvailable: false });
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
      // Remove a GUI container.
      case "GuiRemoveContainerChildrenMessage": {
        removeGuiContainer(message.container_id);
        return;
      }
      default: {
        console.log("Received message did not match any known types:", message);
        return;
      }
    }
  };
}

/** Component for handling websocket connections. */
export default function WebsocketInterface() {
  const viewer = useContext(ViewerContext)!;
  const handleMessage = useMessageHandler();

  const server = viewer.useGui((state) => state.server);
  const resetGui = viewer.useGui((state) => state.resetGui);

  syncSearchParamServer(server);

  const messageQueue: Message[] = [];

  useFrame(() => {
    // Handle messages before every frame.
    // Place this directly in ws.onmessage can cause race conditions!
    const numMessages = messageQueue.length;
    const processBatch = messageQueue.splice(0, numMessages);
    processBatch.forEach(handleMessage);
  });

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

      ws.onclose = () => {
        console.log(`Disconnected! ${server}`);
        clearTimeout(retryTimeout);
        viewer.websocketRef.current = null;
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
          messageQueue.push(...messages);
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
  }, [server, handleMessage, resetGui]);

  return <></>;
}
