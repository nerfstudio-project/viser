// @refresh reset
import {
  AdaptiveDpr,
  AdaptiveEvents,
  CameraControls,
  Environment,
} from "@react-three/drei";
import * as THREE from "three";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import {
  EffectComposer,
  Outline,
  Selection,
} from "@react-three/postprocessing";
import { BlendFunction, KernelSize } from "postprocessing";

import { SynchronizedCameraControls } from "./CameraControls";
import { Box, MantineProvider, MediaQuery } from "@mantine/core";
import React from "react";
import {
  SceneNodeThreeObject,
  UseSceneTree,
} from "./SceneTree";

import "./index.css";

import ControlPanel from "./ControlPanel/ControlPanel";
import { UseGui, useGuiState } from "./ControlPanel/GuiState";
import { searchParamKey } from "./SearchParamsUtils";
import WebsocketInterface from "./WebsocketInterface";

import { Titlebar } from "./Titlebar";
import { useSceneTreeState } from "./SceneTreeState";

type ViewerContextContents = {
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  websocketRef: React.MutableRefObject<WebSocket | null>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  sceneRef: React.MutableRefObject<THREE.Scene | null>;
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>;
  nerfMaterialRef: React.MutableRefObject<THREE.ShaderMaterial | null>;
  cameraControlRef: React.MutableRefObject<CameraControls | null>;
  nodeAttributesFromName: React.MutableRefObject<{
    [name: string]:
      | undefined
      | {
          wxyz?: [number, number, number, number];
          position?: [number, number, number];
          visibility?: boolean;
        };
  }>;
};
export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null
);

THREE.ColorManagement.enabled = true;

function SingleViewer() {
  // Default server logic.
  function getDefaultServerFromUrl() {
    // https://localhost:8080/ => ws://localhost:8080
    // https://localhost:8080/?server=some_url => ws://localhost:8080
    let server = window.location.href;
    server = server.replace("http://", "ws://");
    server = server.split("?")[0];
    if (server.endsWith("/")) server = server.slice(0, -1);
    return server;
  }
  const servers = new URLSearchParams(window.location.search).getAll(
    searchParamKey
  );
  const initialServer =
    servers.length >= 1 ? servers[0] : getDefaultServerFromUrl();

  // Values that can be globally accessed by components in a viewer.
  const viewer: ViewerContextContents = {
    useSceneTree: useSceneTreeState(),
    useGui: useGuiState(initialServer),
    websocketRef: React.useRef(null),
    canvasRef: React.useRef(null),
    sceneRef: React.useRef(null),
    cameraRef: React.useRef(null),
    nerfMaterialRef: React.useRef(null),
    cameraControlRef: React.useRef(null),
    // Scene node attributes that aren't placed in the zustand state, for performance reasons.
    nodeAttributesFromName: React.useRef({}),
  };
  const fixed_sidebar = viewer.useGui((state) => state.theme.fixed_sidebar);
  return (
    <ViewerContext.Provider value={viewer}>
      <Titlebar />
      <Box
        sx={{
          width: "100%",
          height: "1px",
          position: "relative",
          flex: "1 0 auto",
        }}
      >
        <WebsocketInterface />
        <MediaQuery smallerThan={"xs"} styles={{ right: 0, bottom: "3.5em" }}>
          <Box
            sx={(theme) => ({
              top: 0,
              bottom: 0,
              left: 0,
              right: fixed_sidebar ? "20em" : 0,
              position: "absolute",
              backgroundColor:
                theme.colorScheme === "light" ? "#fff" : theme.colors.dark[9],
            })}
          >
            <ViewerCanvas />
          </Box>
        </MediaQuery>
        <ControlPanel fixed_sidebar={fixed_sidebar} />
      </Box>
    </ViewerContext.Provider>
  );
}

function ViewerCanvas() {
  const viewer = React.useContext(ViewerContext)!;
  return (
    <Canvas
      camera={{ position: [3.0, 3.0, -3.0] }}
      gl={{ preserveDrawingBuffer: true }}
      style={{
        position: "relative",
        zIndex: 0,
        width: "100%",
        height: "100%",
      }}
      performance={{ min: 0.95 }}
      ref={viewer.canvasRef}
    >
      <NeRFImage />
      <AdaptiveDpr pixelated />
      <AdaptiveEvents />
      <SceneContextSetter />
      <SynchronizedCameraControls />
      <Selection>
        <SceneNodeThreeObject name="" />
        <EffectComposer enabled={true} autoClear={false}>
          <Outline
            hiddenEdgeColor={0xfbff00}
            visibleEdgeColor={0xfbff00}
            blendFunction={BlendFunction.SCREEN} // set this to BlendFunction.ALPHA for dark outlines
            kernelSize={KernelSize.MEDIUM}
            edgeStrength={30.0}
            height={480}
            blur
          />
        </EffectComposer>
      </Selection>
      <Environment path="/hdri/" files="potsdamer_platz_1k.hdr" />
    </Canvas>
  );
}

function NeRFImage(){
  // Create a fragment shader that composites depth using nerfDepth and nerfColor
  const vertShader = `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
  `.trim();
  const fragShader = `  
  varying vec2 vUv;
  uniform sampler2D nerfColor;
  uniform sampler2D nerfDepth;
  uniform float cameraNear;
  uniform float cameraFar;
  uniform float depthScale;
  uniform bool enabled;

  // depthSample from depthTexture.r, for instance
  float linearDepth(float depthSample, float zNear, float zFar)
  {
      depthSample = 2.0 * depthSample - 1.0;
      float zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear));
      return zLinear;
  }

  // result suitable for assigning to gl_FragDepth
  float depthSample(float linearDepth, float zNear, float zFar)
  {
      float nonLinearDepth = (zFar + zNear - 2.0 * zNear * zFar / linearDepth) / (zFar - zNear);
      nonLinearDepth = (nonLinearDepth + 1.0) / 2.0;
      return nonLinearDepth;
  }

  float readDepth( sampler2D depthSampler, vec2 coord, float zNear, float zFar) {
    float depth = texture(depthSampler,coord).x * depthScale;
    float nonLinearDepth = depthSample(depth, zNear, zFar);
    return nonLinearDepth;
  }

  void main() {
    if (!enabled) {
      // discard the pixel if we're not enabled
      discard;
    }
    vec4 color = texture( nerfColor, vUv );
    gl_FragColor = vec4( color.rgb, 1.0 );

    float depth = readDepth(nerfDepth, vUv, cameraNear, cameraFar);
    if(depth < gl_FragCoord.z){
      gl_FragDepth = depth;
    }else{
      //otherwise set infinite depth
      gl_FragDepth = 1.0;
    }
  }`.trim();
  // initialize the nerfColor texture with all white and depth at infinity
  const nerfMaterial = new THREE.ShaderMaterial({
    fragmentShader: fragShader,
    vertexShader: vertShader,
    uniforms: {
      enabled: {value: false},
      nerfDepth: {value: null},
      depthScale: {value: 1.0},
      nerfColor: {value: null},
      cameraNear: {value: null},
      cameraFar: {value: null},
    }
  });
  const { nerfMaterialRef } = React.useContext(ViewerContext)!;
  nerfMaterialRef.current = nerfMaterial;
  const nerfMesh = React.useRef<THREE.Mesh>(null);
  useFrame(({camera}) => {
    //assert it is a perspective camera
    if(!(camera instanceof THREE.PerspectiveCamera)){
      console.error("Camera is not a perspective camera, cannot render NeRF image");
      return;
    }
    // Update the position of the mesh based on the camera position
    const lookdir = camera.getWorldDirection(new THREE.Vector3());
    nerfMesh.current!.position.set(camera.position.x,camera.position.y,camera.position.z);
    nerfMesh.current!.position.addScaledVector(lookdir,1.0);
    nerfMesh.current!.quaternion.copy(camera.quaternion);
    //resize the mesh based on size
    const f = camera.getFocalLength();
    nerfMesh.current!.scale.set(camera.getFilmWidth()/f,camera.getFilmHeight()/f,1.0);
    //set the near/far uniforms
    nerfMaterial.uniforms.cameraNear.value = camera.near;
    nerfMaterial.uniforms.cameraFar.value = camera.far;
  });
  return <mesh
            ref={nerfMesh}
            material={nerfMaterial}
            matrixWorldAutoUpdate={false}
            name="nerfImage"
          >
            <planeGeometry
              attach="geometry"
              args={[1 , 1]}
            />
          </mesh>
}

/** Component for helping us set the scene reference. */
function SceneContextSetter() {
  const { sceneRef, cameraRef } = React.useContext(ViewerContext)!;
  sceneRef.current = useThree((state) => state.scene);
  cameraRef.current = useThree(
    (state) => state.camera as THREE.PerspectiveCamera
  );
  return <></>;
}

export function Root() {
  return (
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
      theme={{
        colorScheme: "light",
      }}
    >
      <Box
        sx={{
          width: "100%",
          height: "100%",
          position: "relative",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <SingleViewer />
      </Box>
    </MantineProvider>
  );
}
