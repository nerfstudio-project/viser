import { pack, unpack } from "msgpackr";
import React, { MutableRefObject, RefObject } from "react";
import * as THREE from "three";
import AwaitLock from "await-lock";
import { TextureLoader } from "three";
import { UseGui } from "./ControlPanel/GuiState";

import { SceneNode, UseSceneTree } from "./SceneTree";
import { CoordinateFrame, CameraFrustum } from "./ThreeAssets";
import { Message } from "./WebsocketMessages";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { PivotControls } from "@react-three/drei";

/** Send message over websocket. */
export function sendWebsocketMessage(
  websocketRef: MutableRefObject<WebSocket | null>,
  message: Message
) {
  if (websocketRef.current === null) return;
  websocketRef.current.send(pack(message));
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  websocketRef: MutableRefObject<WebSocket | null>,
  throttleMilliseconds: number
) {
  let readyToSend = true;
  let stale = false;
  let latestMessage: Message | null = null;

  function send(message: Message) {
    if (websocketRef.current === null) return;
    latestMessage = message;
    if (readyToSend) {
      websocketRef.current.send(pack(message));
      stale = false;
      readyToSend = false;

      setTimeout(() => {
        readyToSend = true;
        if (!stale) return;
        latestMessage && send(latestMessage);
      }, throttleMilliseconds);
    } else {
      stale = true;
    }
  }
  return send;
}

/** Returns a handler for all incoming messages. */
function useMessageHandler(
  useSceneTree: UseSceneTree,
  useGui: UseGui,
  wrapperRef: RefObject<HTMLDivElement>,
  websocketRef: MutableRefObject<WebSocket | null>
) {
  const removeSceneNode = useSceneTree((state) => state.removeSceneNode);
  const resetScene = useSceneTree((state) => state.resetScene);
  const addSceneNode = useSceneTree((state) => state.addSceneNode);
  const addGui = useGui((state) => state.addGui);
  const removeGui = useGui((state) => state.removeGui);
  const guiSet = useGui((state) => state.guiSet);
  const setVisibility = useSceneTree((state) => state.setVisibility);
  const setMatrix = useSceneTree((state) => state.setMatrix);

  // Same as addSceneNode, but make a parent in the form of a dummy coordinate
  // frame if it doesn't exist yet.
  function addSceneNodeMakeParents(node: SceneNode) {
    const nodeFromName = useSceneTree.getState().nodeFromName;
    const parent_name = node.name.split("/").slice(0, -1).join("/");
    if (!(parent_name in nodeFromName)) {
      addSceneNodeMakeParents(
        new SceneNode(parent_name, (ref) => (
          <CoordinateFrame ref={ref} show_axes={false} />
        ))
      );
    }
    addSceneNode(node);
  }

  // Return message handler.
  return (message: Message) => {
    switch (message.type) {
      // Add a coordinate frame.
      case "FrameMessage": {
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <CoordinateFrame
              ref={ref}
              position={new THREE.Vector3().fromArray(message.position)}
              quaternion={
                new THREE.Quaternion(
                  message.wxyz[1],
                  message.wxyz[2],
                  message.wxyz[3],
                  message.wxyz[0]
                )
              }
              show_axes={message.show_axes}
              axes_length={message.axes_length}
              axes_radius={message.axes_radius}
            />
          ))
        );
        break;
      }
      // Add a point cloud.
      case "PointCloudMessage": {
        const geometry = new THREE.BufferGeometry();
        const pointCloudMaterial = new THREE.PointsMaterial({
          size: message.point_size,
          vertexColors: true,
        });

        // Reinterpret cast: uint8 buffer => float32 for positions.
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.position.buffer.slice(
                message.position.byteOffset,
                message.position.byteOffset + message.position.byteLength
              )
            ),
            3
          )
        );
        geometry.computeBoundingSphere();

        // Wrap uint8 buffer for colors. Note that we need to set normalized=true.
        geometry.setAttribute(
          "color",
          new THREE.Uint8BufferAttribute(message.color, 3, true)
        );

        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <points
              ref={ref}
              geometry={geometry}
              material={pointCloudMaterial}
            />
          ))
        );
        break;
      }
      // Add mesh
      case "MeshMessage": {
        const geometry = new THREE.BufferGeometry();
        // TODO(hangg): Should expose color as well.
        const material = new THREE.MeshStandardMaterial({
          color: message.color,
          wireframe: message.wireframe,
        });
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.vertices.buffer.slice(
                message.vertices.byteOffset,
                message.vertices.byteOffset + message.vertices.byteLength
              )
            ),
            3
          )
        );
        geometry.setIndex(
          new THREE.Uint32BufferAttribute(
            new Uint32Array(
              message.faces.buffer.slice(
                message.faces.byteOffset,
                message.faces.byteOffset + message.faces.byteLength
              )
            ),
            1
          )
        );
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <mesh ref={ref} geometry={geometry} material={material} />
          ))
        );
        break;
      }
      // Add a camera frustum.
      case "CameraFrustumMessage": {
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <CameraFrustum
              ref={ref}
              fov={message.fov}
              aspect={message.aspect}
              scale={message.scale}
              color={message.color}
            ></CameraFrustum>
          ))
        );
        break;
      }
      case "TransformControlsMessage": {
        const name = message.name;
        const sendDragMessage = makeThrottledMessageSender(websocketRef, 25);
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
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
                const wxyz = new THREE.Quaternion();
                wxyz.setFromRotationMatrix(l);
                const position = new THREE.Vector3().setFromMatrixPosition(l);
                sendDragMessage({
                  type: "TransformControlsUpdateMessage",
                  name: name,
                  wxyz: [wxyz.w, wxyz.x, wxyz.y, wxyz.z],
                  position: position.toArray(),
                });
              }}
            />
          ))
        );
        break;
      }
      case "SetTransformMessage": {
        setMatrix(
          message.name,
          new THREE.Matrix4()
            .makeRotationFromQuaternion(
              new THREE.Quaternion(
                message.wxyz[1],
                message.wxyz[2],
                message.wxyz[3],
                message.wxyz[0]
              )
            )
            .setPosition(
              message.position[0],
              message.position[1],
              message.position[2]
            )
        );
        break;
      }
      // Add a background image.
      case "BackgroundImageMessage": {
        if (wrapperRef.current != null) {
          wrapperRef.current.style.backgroundImage = `url(data:${message.media_type};base64,${message.base64_data})`;
          wrapperRef.current.style.backgroundSize = "cover";
          wrapperRef.current.style.backgroundRepeat = "no-repeat";
          wrapperRef.current.style.backgroundPosition = "center center";

          useGui.setState({ backgroundAvailable: true });
        }
        break;
      }
      // Add an image.
      case "ImageMessage": {
        // It's important that we load the texture outside of the node
        // construction callback; this prevents flickering by ensuring that the
        // texture is ready before the scene tree updates.
        const colorMap = new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_data}`
        );
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => {
            return (
              <mesh ref={ref}>
                <planeGeometry
                  attach="geometry"
                  args={[message.render_width, message.render_height]}
                />
                <meshBasicMaterial
                  attach="material"
                  transparent={true}
                  side={THREE.DoubleSide}
                  map={colorMap}
                />
              </mesh>
            );
          })
        );
        break;
      }
      // Remove a scene node by name.
      case "RemoveSceneNodeMessage": {
        console.log("Removing scene node:", message.name);
        removeSceneNode(message.name);
        break;
      }
      // Set the visibility of a particular scene node.
      case "SetSceneNodeVisibilityMessage": {
        setVisibility(message.name, message.visible);
        break;
      }
      // Reset the entire scene, removing all scene nodes.
      case "ResetSceneMessage": {
        resetScene();
        wrapperRef.current &&
          (wrapperRef.current.style.backgroundImage = "none");

        useGui.setState({ backgroundAvailable: false });
        break;
      }
      // Add a GUI input.
      case "GuiAddMessage": {
        addGui(message.name, {
          levaConf: message.leva_conf,
          folderLabels: message.folder_labels,
          hidden: false,
        });
        break;
      }
      // Set the value of a GUI input.
      case "GuiSetValueMessage": {
        guiSet(message.name, message.value);
        break;
      }
      // Set the hidden state of a GUI input.
      case "GuiSetHiddenMessage": {
        const currentConf = useGui.getState().guiConfigFromName[message.name];
        if (currentConf !== undefined) {
          addGui(message.name, {
            ...currentConf,
            hidden: message.hidden,
          });
        }
        break;
      }
      // Add a GUI input.
      case "GuiSetLevaConfMessage": {
        const currentConf = useGui.getState().guiConfigFromName[message.name];
        if (currentConf !== undefined) {
          addGui(message.name, {
            ...currentConf,
            levaConf: message.leva_conf,
          });
        }
        break;
      }
      // Remove a GUI input.
      case "GuiRemoveMessage": {
        removeGui(message.name);
        break;
      }
      default: {
        console.log("Received message did not match any known types:", message);
        break;
      }
    }
  };
}

/** Component for handling websocket connections. */
export default function WebsocketInterface(props: {
  panelKey: number;
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  websocketRef: MutableRefObject<WebSocket | null>;
  wrapperRef: RefObject<HTMLDivElement>;
}) {
  const handleMessage = useMessageHandler(
    props.useSceneTree,
    props.useGui,
    props.wrapperRef,
    props.websocketRef
  );

  const server = props.useGui((state) => state.server);
  const resetGui = props.useGui((state) => state.resetGui);

  syncSearchParamServer(props.panelKey, server);

  React.useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let ws: null | WebSocket = null;
    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws = new WebSocket(server);

      ws.onopen = () => {
        console.log("Connected!" + server);
        props.websocketRef.current = ws;
        props.useGui.setState({ websocketConnected: true });
      };

      ws.onclose = () => {
        console.log("Disconnected! " + server);
        props.websocketRef.current = null;
        props.useGui.setState({ websocketConnected: false });
        if (props.useGui.getState().guiNames.length > 0) resetGui();

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      const messageQueue: Message[] = [];

      // Handle batches of messages at 200Hz. This helps prevent some React errors from interrupting renders.
      setInterval(() => {
        const numMessages = messageQueue.length;
        const processBatch = messageQueue.slice(0, numMessages);
        messageQueue.splice(0, numMessages);
        processBatch.forEach(handleMessage);
      }, 5);

      ws.onmessage = async (event) => {
        // Reduce websocket backpressure.
        const messagePromise = new Promise<Message>((resolve) => {
          (event.data.arrayBuffer() as Promise<ArrayBuffer>).then((buffer) => {
            resolve(unpack(new Uint8Array(buffer)) as Message);
          });
        });

        // Try our best to handle messages in order. If this takes more than 1 second, we give up. :)
        await orderLock.acquireAsync({ timeout: 1000 }).catch(() => {
          console.log("Order lock timed out.");
          orderLock.release();
        });
        try {
          messageQueue.push(await messagePromise);
        } finally {
          orderLock.acquired && orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      props.useGui.setState({ websocketConnected: false });
      ws && ws.close();
      clearTimeout(timeout);
    };
  }, [props, server, handleMessage, resetGui]);

  return <></>;
}
