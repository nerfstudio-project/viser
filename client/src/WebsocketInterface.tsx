import { decode } from "@msgpack/msgpack";
import AwaitLock from "await-lock";
import React, { MutableRefObject, RefObject } from "react";
import * as THREE from "three";
import { TextureLoader } from "three";
import { UseGui } from "./GuiState";

import { SceneNode, UseSceneTree } from "./SceneTree";
import { CoordinateFrame, CameraFrustum } from "./ThreeAssets";
import { Message } from "./WebsocketMessages";

/** React hook for handling incoming messages, and using them for scene tree manipulation. */
function useMessageHandler(
  useSceneTree: UseSceneTree,
  useGui: UseGui,
  wrapperRef: RefObject<HTMLDivElement>
) {
  const removeSceneNode = useSceneTree((state) => state.removeSceneNode);
  const resetScene = useSceneTree((state) => state.resetScene);
  const addSceneNode = useSceneTree((state) => state.addSceneNode);
  const addGui = useGui((state) => state.addGui);

  // Return message handler.
  return (message: Message) => {
    switch (message.type) {
      // Add a coordinate frame.
      case "frame": {
        addSceneNode(
          new SceneNode(message.name, (ref) => (
            <CoordinateFrame
              ref={ref}
              scale={message.scale}
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
            />
          ))
        );
        break;
      }
      // Add a point cloud.
      case "point_cloud": {
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
              message.position_f32.buffer.slice(
                message.position_f32.byteOffset,
                message.position_f32.byteOffset +
                  message.position_f32.byteLength
              )
            ),
            3
          )
        );
        geometry.computeBoundingSphere();

        // Wrap uint8 buffer for colors. Note that we need to set normalized=true.
        geometry.setAttribute(
          "color",
          new THREE.Uint8BufferAttribute(message.color_uint8, 3, true)
        );

        addSceneNode(
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
      case "mesh": {
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.MeshStandardMaterial({ color: 0x00e8fc });
        geometry.setAttribute(
          "position",
           new THREE.Float32BufferAttribute(
            new Float32Array(
              message.vertices_f32.buffer.slice(
                message.vertices_f32.byteOffset,
                message.vertices_f32.byteOffset + message.vertices_f32.byteLength
              )
            ),
            3
          )
        );
        geometry.setIndex(
          new THREE.Uint32BufferAttribute(
            new Uint32Array(
              message.faces_uint32.buffer.slice(
                message.faces_uint32.byteOffset,
                message.faces_uint32.byteOffset + message.faces_uint32.byteLength
              )
            ),
            1
          )
        );
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        addSceneNode(
          new SceneNode(message.name, (ref) => (
            <mesh
              ref={ref}
              geometry={geometry}
              material={material}
            />
          ))
        );

        addSceneNode(
          new SceneNode("light", (ref) => (
            <group ref={ref}>
              <pointLight position={ [-10, -10, -10] } />;
              <pointLight position={ [10, 10, 10] } />;
              <ambientLight intensity={ 0.2 } />
            </group>
          ))
        );
        break;
      }

      // Add a camera frustum.
      case "camera_frustum": {
        addSceneNode(
          new SceneNode(message.name, (ref) => (
            <CameraFrustum
              ref={ref}
              fov={message.fov}
              aspect={message.aspect}
              scale={message.scale}
            ></CameraFrustum>
          ))
        );
        break;
      }
      // Add a background image.
      case "background_image": {
        if (wrapperRef.current != null) {
          wrapperRef.current.style.backgroundImage = `url(data:${message.media_type};base64,${message.base64_data})`;
          wrapperRef.current.style.backgroundSize = "cover";
          wrapperRef.current.style.backgroundRepeat = "no-repeat";
          wrapperRef.current.style.backgroundPosition = "center center";
        }
        break;
      }
      // Add an image.
      case "image": {
        // It's important that we load the texture outside of the node
        // construction callback; this prevents flickering by ensuring that the
        // texture is ready before the scene tree updates.
        const colorMap = new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_data}`
        );
        addSceneNode(
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
      case "remove_scene_node": {
        console.log("Removing scene node:", message.name);
        removeSceneNode(message.name);
        break;
      }
      // Reset the entire scene, removing all scene nodes.
      case "reset_scene": {
        console.log("Resetting scene!");
        resetScene();
        wrapperRef.current!.style.backgroundImage = "none";
        break;
      }
      // Add a GUI input.
      case "add_gui": {
        addGui(message.name, message.leva_conf);
        break;
      }
      default: {
        console.log("Received message did not match any known types:", message);
        break;
      }
    }
  };
}

interface WebSocketInterfaceProps {
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  websocketRef: MutableRefObject<WebSocket | null>;
  wrapperRef: RefObject<HTMLDivElement>;
}

/** Component for handling websocket connections. */
export default function WebsocketInterface(props: WebSocketInterfaceProps) {
  const handleMessage = useMessageHandler(
    props.useSceneTree,
    props.useGui,
    props.wrapperRef
  );

  const server = props.useGui((state) => state.server);
  const setWebsocketConnected = props.useGui(
    (state) => state.setWebsocketConnected
  );
  const resetGui = props.useGui((state) => state.resetGui);

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
        setWebsocketConnected(true);
      };

      ws.onclose = () => {
        console.log("Disconnected! " + server);
        props.websocketRef.current = null;
        setWebsocketConnected(false);
        resetGui();

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      ws.onmessage = async (event) => {
        // Async message handler. This is structured to reduce websocket
        // backpressure.
        const messagePromise = new Promise<Message>(async (resolve) => {
          resolve(decode(await event.data.arrayBuffer()) as Message);
        });

        // Handle messages in order.
        await orderLock.acquireAsync();
        try {
          handleMessage(await messagePromise);
        } finally {
          orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      setWebsocketConnected(false);
      ws && ws.close();
      clearTimeout(timeout);
    };
  }, [props, server, setWebsocketConnected]);

  return <></>;
}
