import { Instance, Instances, Line, shaderMaterial } from "@react-three/drei";
import { createPortal, useFrame, useThree } from "@react-three/fiber";
import { Outlines } from "./Outlines";
import React from "react";
import * as THREE from "three";
import { GLTF, GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import {
  MeshBasicMaterial,
  MeshDepthMaterial,
  MeshDistanceMaterial,
  MeshLambertMaterial,
  MeshMatcapMaterial,
  MeshNormalMaterial,
  MeshPhongMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  MeshToonMaterial,
  ShadowMaterial,
  SpriteMaterial,
  RawShaderMaterial,
  ShaderMaterial,
  PointsMaterial,
  LineBasicMaterial,
  LineDashedMaterial,
} from "three";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader";
import {
  ImageMessage,
  MeshMessage,
  SkinnedMeshMessage,
} from "./WebsocketMessages";
import { ViewerContext } from "./App";

type AllPossibleThreeJSMaterials =
  | MeshBasicMaterial
  | MeshDepthMaterial
  | MeshDistanceMaterial
  | MeshLambertMaterial
  | MeshMatcapMaterial
  | MeshNormalMaterial
  | MeshPhongMaterial
  | MeshPhysicalMaterial
  | MeshStandardMaterial
  | MeshToonMaterial
  | ShadowMaterial
  | SpriteMaterial
  | RawShaderMaterial
  | ShaderMaterial
  | PointsMaterial
  | LineBasicMaterial
  | LineDashedMaterial;

function rgbToInt(rgb: [number, number, number]): number {
  return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
}
const originGeom = new THREE.SphereGeometry(1.0);

const PointCloudMaterial = /* @__PURE__ */ shaderMaterial(
  { scale: 1.0, point_ball_norm: 0.0 },
  `
  precision mediump float;

  varying vec3 vPosition;
  varying vec3 vColor; // in the vertex shader
  uniform float scale;

  void main() {
      vPosition = position;
      vColor = color;
      vec4 world_pos = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * world_pos;
      gl_PointSize = (scale / -world_pos.z);
  }
   `,
  `varying vec3 vPosition;
  varying vec3 vColor;
  uniform float point_ball_norm;

  void main() {
      if (point_ball_norm < 1000.0) {
          float r = pow(
              pow(abs(gl_PointCoord.x - 0.5), point_ball_norm)
              + pow(abs(gl_PointCoord.y - 0.5), point_ball_norm),
              1.0 / point_ball_norm);
          if (r > 0.5) discard;
      }
      gl_FragColor = vec4(vColor, 1.0);
  }
   `,
);

export const PointCloud = React.forwardRef<
  THREE.Points,
  {
    pointSize: number;
    /** We visualize each point as a 2D ball, which is defined by some norm. */
    pointBallNorm: number;
    points: Uint16Array; // Contains float16.
    colors: Uint8Array;
  }
>(function PointCloud(props, ref) {
  const getThreeState = useThree((state) => state.get);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float16BufferAttribute(props.points, 3),
  );
  geometry.computeBoundingSphere();
  geometry.setAttribute(
    "color",
    new THREE.BufferAttribute(props.colors, 3, true),
  );

  const [material] = React.useState(
    () => new PointCloudMaterial({ vertexColors: true }),
  );
  material.uniforms.scale.value = 10.0;
  material.uniforms.point_ball_norm.value = props.pointBallNorm;

  React.useEffect(() => {
    return () => {
      material.dispose();
      geometry.dispose();
    };
  });

  const rendererSize = new THREE.Vector2();
  useFrame(() => {
    // Match point scale to behavior of THREE.PointsMaterial().
    if (material === undefined) return;
    // point px height / actual height = point meters height / frustum meters height
    // frustum meters height = math.tan(fov / 2.0) * z
    // point px height = (point meters height / math.tan(fov / 2.0) * actual height)  / z
    material.uniforms.scale.value =
      (props.pointSize /
        Math.tan(
          (((getThreeState().camera as THREE.PerspectiveCamera).fov / 180.0) *
            Math.PI) /
            2.0,
        )) *
      getThreeState().gl.getSize(rendererSize).height *
      getThreeState().gl.getPixelRatio();
  });
  return <points ref={ref} geometry={geometry} material={material} />;
});

/** Component for rendering the contents of GLB files. */
export const GlbAsset = React.forwardRef<
  THREE.Group,
  { glb_data: Uint8Array; scale: number }
>(function GlbAsset({ glb_data, scale }, ref) {
  // We track both the GLTF asset itself and all meshes within it. Meshes are
  // used for hover effects.
  const [gltf, setGltf] = React.useState<GLTF>();
  const [meshes, setMeshes] = React.useState<THREE.Mesh[]>([]);

  // glTF/GLB files support animations.
  const mixerRef = React.useRef<THREE.AnimationMixer | null>(null);

  React.useEffect(() => {
    const loader = new GLTFLoader();

    // We use a CDN for Draco. We could move this locally if we want to use Viser offline.
    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath("https://www.gstatic.com/draco/v1/decoders/");
    loader.setDRACOLoader(dracoLoader);

    loader.parse(
      glb_data.buffer,
      "",
      (gltf) => {
        if (gltf.animations && gltf.animations.length) {
          mixerRef.current = new THREE.AnimationMixer(gltf.scene);
          gltf.animations.forEach((clip) => {
            mixerRef.current!.clipAction(clip).play();
          });
        }
        const meshes: THREE.Mesh[] = [];
        gltf?.scene.traverse((obj) => {
          if (obj instanceof THREE.Mesh) meshes.push(obj);
        });
        setMeshes(meshes);
        setGltf(gltf);
      },
      (error) => {
        console.log("Error loading GLB!");
        console.log(error);
      },
    );

    return () => {
      if (mixerRef.current) mixerRef.current.stopAllAction();

      function disposeNode(node: any) {
        if (node instanceof THREE.Mesh) {
          if (node.geometry) {
            node.geometry.dispose();
          }
          if (node.material) {
            if (Array.isArray(node.material)) {
              node.material.forEach((material) => {
                disposeMaterial(material);
              });
            } else {
              disposeMaterial(node.material);
            }
          }
        }
      }
      function disposeMaterial(material: AllPossibleThreeJSMaterials) {
        if ("map" in material) material.map?.dispose();
        if ("lightMap" in material) material.lightMap?.dispose();
        if ("bumpMap" in material) material.bumpMap?.dispose();
        if ("normalMap" in material) material.normalMap?.dispose();
        if ("specularMap" in material) material.specularMap?.dispose();
        if ("envMap" in material) material.envMap?.dispose();
        if ("alphaMap" in material) material.alphaMap?.dispose();
        if ("aoMap" in material) material.aoMap?.dispose();
        if ("displacementMap" in material) material.displacementMap?.dispose();
        if ("emissiveMap" in material) material.emissiveMap?.dispose();
        if ("gradientMap" in material) material.gradientMap?.dispose();
        if ("metalnessMap" in material) material.metalnessMap?.dispose();
        if ("roughnessMap" in material) material.roughnessMap?.dispose();
        material.dispose(); // disposes any programs associated with the material
      }

      // Attempt to free resources.
      gltf?.scene.traverse(disposeNode);
    };
  }, [glb_data]);

  useFrame((_, delta) => {
    if (mixerRef.current) {
      mixerRef.current.update(delta);
    }
  });

  return (
    <group ref={ref}>
      {gltf === undefined ? null : (
        <>
          <primitive object={gltf.scene} scale={scale} />
          {meshes.map((mesh) =>
            createPortal(<OutlinesIfHovered alwaysMounted />, mesh),
          )}
        </>
      )}
    </group>
  );
});

/** Helper for adding coordinate frames as scene nodes. */
export const CoordinateFrame = React.forwardRef<
  THREE.Group,
  {
    showAxes?: boolean;
    axesLength?: number;
    axesRadius?: number;
    originRadius?: number;
    originColor?: number;
  }
>(function CoordinateFrame(
  {
    showAxes = true,
    axesLength = 0.5,
    axesRadius = 0.0125,
    originRadius = undefined,
    originColor = 0xecec00,
  },
  ref,
) {
  originRadius = originRadius ?? axesRadius * 2;
  return (
    <group ref={ref}>
      {showAxes && (
        <>
          <mesh
            geometry={originGeom}
            scale={new THREE.Vector3(originRadius, originRadius, originRadius)}
          >
            <meshBasicMaterial color={originColor} />
            <OutlinesIfHovered />
          </mesh>
          <Instances limit={3}>
            <meshBasicMaterial />
            <cylinderGeometry args={[axesRadius, axesRadius, axesLength, 16]} />
            <Instance
              rotation={new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0)}
              position={[0.5 * axesLength, 0.0, 0.0]}
              color={0xcc0000}
            >
              <OutlinesIfHovered />
            </Instance>
            <Instance position={[0.0, 0.5 * axesLength, 0.0]} color={0x00cc00}>
              <OutlinesIfHovered />
            </Instance>
            <Instance
              rotation={new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)}
              position={[0.0, 0.0, 0.5 * axesLength]}
              color={0x0000cc}
            >
              <OutlinesIfHovered />
            </Instance>
          </Instances>
        </>
      )}
    </group>
  );
});

/** Helper for adding batched/instanced coordinate frames as scene nodes. */
export const InstancedAxes = React.forwardRef<
  THREE.Group,
  {
    wxyzsBatched: Float32Array;
    positionsBatched: Float32Array;
    axes_length?: number;
    axes_radius?: number;
  }
>(function InstancedAxes(
  {
    wxyzsBatched: instance_wxyzs,
    positionsBatched: instance_positions,
    axes_length = 0.5,
    axes_radius = 0.0125,
  },
  ref,
) {
  const axesRef = React.useRef<THREE.InstancedMesh>(null);

  const cylinderGeom = new THREE.CylinderGeometry(
    axes_radius,
    axes_radius,
    axes_length,
    16,
  );
  const material = new MeshBasicMaterial();

  // Dispose when done.
  React.useEffect(() => {
    return () => {
      cylinderGeom.dispose();
      material.dispose();
    };
  });

  // Update instance matrices and colors.
  React.useEffect(() => {
    // Pre-allocate to avoid garbage collector from running during loop.
    const T_world_frame = new THREE.Matrix4();
    const T_world_framex = new THREE.Matrix4();
    const T_world_framey = new THREE.Matrix4();
    const T_world_framez = new THREE.Matrix4();

    const T_frame_framex = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0))
      .setPosition(0.5 * axes_length, 0.0, 0.0);
    const T_frame_framey = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(0.0, 0.0, 0.0))
      .setPosition(0.0, 0.5 * axes_length, 0.0);
    const T_frame_framez = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(Math.PI / 2.0, 0.0, 0.0))
      .setPosition(0.0, 0.0, 0.5 * axes_length);

    const tmpQuat = new THREE.Quaternion();

    const red = new THREE.Color(0xcc0000);
    const green = new THREE.Color(0x00cc00);
    const blue = new THREE.Color(0x0000cc);

    for (let i = 0; i < instance_wxyzs.length / 4; i++) {
      T_world_frame.makeRotationFromQuaternion(
        tmpQuat.set(
          instance_wxyzs[i * 4 + 1],
          instance_wxyzs[i * 4 + 2],
          instance_wxyzs[i * 4 + 3],
          instance_wxyzs[i * 4 + 0],
        ),
      ).setPosition(
        instance_positions[i * 3 + 0],
        instance_positions[i * 3 + 1],
        instance_positions[i * 3 + 2],
      );
      T_world_framex.copy(T_world_frame).multiply(T_frame_framex);
      T_world_framey.copy(T_world_frame).multiply(T_frame_framey);
      T_world_framez.copy(T_world_frame).multiply(T_frame_framez);

      axesRef.current!.setMatrixAt(i * 3 + 0, T_world_framex);
      axesRef.current!.setMatrixAt(i * 3 + 1, T_world_framey);
      axesRef.current!.setMatrixAt(i * 3 + 2, T_world_framez);

      axesRef.current!.setColorAt(i * 3 + 0, red);
      axesRef.current!.setColorAt(i * 3 + 1, green);
      axesRef.current!.setColorAt(i * 3 + 2, blue);
    }
    axesRef.current!.instanceMatrix.needsUpdate = true;
    axesRef.current!.instanceColor!.needsUpdate = true;
  }, [instance_wxyzs, instance_positions]);

  return (
    <group ref={ref}>
      <instancedMesh
        ref={axesRef}
        args={[cylinderGeom, material, (instance_wxyzs.length / 4) * 3]}
      >
        <OutlinesIfHovered />
      </instancedMesh>
    </group>
  );
});

/** Convert raw RGB color buffers to linear color buffers. **/
export const ViserMesh = React.forwardRef<
  THREE.Mesh | THREE.SkinnedMesh,
  MeshMessage | SkinnedMeshMessage
>(function ViserMesh(message, ref) {
  const viewer = React.useContext(ViewerContext)!;

  const generateGradientMap = (shades: 3 | 5) => {
    const texture = new THREE.DataTexture(
      Uint8Array.from(
        shades == 3
          ? [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]
          : [
              0, 0, 0, 255, 64, 64, 64, 255, 128, 128, 128, 255, 192, 192, 192,
              255, 255, 255, 255, 255,
            ],
      ),
      shades,
      1,
      THREE.RGBAFormat,
    );

    texture.needsUpdate = true;
    return texture;
  };
  const standardArgs = {
    color:
      message.props.color === null ? undefined : rgbToInt(message.props.color),
    wireframe: message.props.wireframe,
    transparent: message.props.opacity !== null,
    opacity: message.props.opacity ?? 1.0,
    // Flat shading only makes sense for non-wireframe materials.
    flatShading: message.props.flat_shading && !message.props.wireframe,
    side: {
      front: THREE.FrontSide,
      back: THREE.BackSide,
      double: THREE.DoubleSide,
    }[message.props.side],
  };
  const assertUnreachable = (x: never): never => {
    throw new Error(`Should never get here! ${x}`);
  };

  const [material, setMaterial] = React.useState<THREE.Material>();
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [skeleton, setSkeleton] = React.useState<THREE.Skeleton>();
  const bonesRef = React.useRef<THREE.Bone[]>();

  React.useEffect(() => {
    const material =
      message.props.material == "standard" || message.props.wireframe
        ? new THREE.MeshStandardMaterial(standardArgs)
        : message.props.material == "toon3"
          ? new THREE.MeshToonMaterial({
              gradientMap: generateGradientMap(3),
              ...standardArgs,
            })
          : message.props.material == "toon5"
            ? new THREE.MeshToonMaterial({
                gradientMap: generateGradientMap(5),
                ...standardArgs,
              })
            : assertUnreachable(message.props.material);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(
          message.props.vertices.buffer.slice(
            message.props.vertices.byteOffset,
            message.props.vertices.byteOffset +
              message.props.vertices.byteLength,
          ),
        ),
        3,
      ),
    );

    geometry.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(
          message.props.faces.buffer.slice(
            message.props.faces.byteOffset,
            message.props.faces.byteOffset + message.props.faces.byteLength,
          ),
        ),
        1,
      ),
    );
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();

    let skeleton = undefined;
    if (message.type === "SkinnedMeshMessage") {
      // Skinned mesh.
      const bone_wxyzs = new Float32Array(
        message.props.bone_wxyzs.buffer.slice(
          message.props.bone_wxyzs.byteOffset,
          message.props.bone_wxyzs.byteOffset +
            message.props.bone_wxyzs.byteLength,
        ),
      );
      const bone_positions = new Float32Array(
        message.props.bone_positions.buffer.slice(
          message.props.bone_positions.byteOffset,
          message.props.bone_positions.byteOffset +
            message.props.bone_positions.byteLength,
        ),
      );

      const bones: THREE.Bone[] = [];
      bonesRef.current = bones;
      for (let i = 0; i < bone_positions.length / 3; i++) {
        bones.push(new THREE.Bone());
      }
      const boneInverses: THREE.Matrix4[] = [];
      const xyzw_quat = new THREE.Quaternion();
      bones.forEach((bone, i) => {
        xyzw_quat.set(
          bone_wxyzs[i * 4 + 1],
          bone_wxyzs[i * 4 + 2],
          bone_wxyzs[i * 4 + 3],
          bone_wxyzs[i * 4 + 0],
        );

        const boneInverse = new THREE.Matrix4();
        boneInverse.makeRotationFromQuaternion(xyzw_quat);
        boneInverse.setPosition(
          bone_positions[i * 3 + 0],
          bone_positions[i * 3 + 1],
          bone_positions[i * 3 + 2],
        );
        boneInverse.invert();
        boneInverses.push(boneInverse);

        bone.quaternion.copy(xyzw_quat);
        bone.position.set(
          bone_positions[i * 3 + 0],
          bone_positions[i * 3 + 1],
          bone_positions[i * 3 + 2],
        );
      });
      skeleton = new THREE.Skeleton(bones, boneInverses);

      geometry.setAttribute(
        "skinIndex",
        new THREE.BufferAttribute(
          new Uint16Array(
            message.props.skin_indices.buffer.slice(
              message.props.skin_indices.byteOffset,
              message.props.skin_indices.byteOffset +
                message.props.skin_indices.byteLength,
            ),
          ),
          4,
        ),
      );
      geometry.setAttribute(
        "skinWeight",
        new THREE.BufferAttribute(
          new Float32Array(
            message.props.skin_weights!.buffer.slice(
              message.props.skin_weights!.byteOffset,
              message.props.skin_weights!.byteOffset +
                message.props.skin_weights!.byteLength,
            ),
          ),
          4,
        ),
      );
    }

    setMaterial(material);
    setGeometry(geometry);
    setSkeleton(skeleton);
    return () => {
      // TODO: we can switch to the react-three-fiber <bufferGeometry />,
      // <meshStandardMaterial />, etc components to avoid manual
      // disposal.
      geometry.dispose();
      material.dispose();

      if (message.type === "SkinnedMeshMessage") {
        skeleton !== undefined && skeleton.dispose();
        const state = viewer.skinnedMeshState.current[message.name];
        state.initialized = false;
      }
    };
  }, [message]);

  // Update bone transforms for skinned meshes.
  useFrame(() => {
    if (message.type !== "SkinnedMeshMessage") return;

    const parentNode = viewer.nodeRefFromName.current[message.name];
    if (parentNode === undefined) return;

    const state = viewer.skinnedMeshState.current[message.name];
    const bones = bonesRef.current;
    if (bones !== undefined) {
      if (!state.initialized) {
        bones.forEach((bone) => {
          parentNode.add(bone);
        });
        state.initialized = true;
      }
      bones.forEach((bone, i) => {
        const wxyz = state.poses[i].wxyz;
        const position = state.poses[i].position;
        bone.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
        bone.position.set(position[0], position[1], position[2]);
      });
    }
  });

  if (geometry === undefined || material === undefined) {
    return;
  } else if (message.type === "SkinnedMeshMessage") {
    return (
      <skinnedMesh
        ref={ref as React.ForwardedRef<THREE.SkinnedMesh>}
        geometry={geometry}
        material={material}
        skeleton={skeleton}
        // TODO: leaving culling on (default) sometimes causes the
        // mesh to randomly disappear, as of r3f==8.16.2.
        //
        // Probably this is because we don't update the bounding
        // sphere after the bone transforms change.
        frustumCulled={false}
      >
        <OutlinesIfHovered alwaysMounted />
      </skinnedMesh>
    );
  } else {
    // Normal mesh.
    return (
      <mesh
        ref={ref as React.ForwardedRef<THREE.Mesh>}
        geometry={geometry}
        material={material}
      >
        <OutlinesIfHovered alwaysMounted />
      </mesh>
    );
  }
});

export const ViserImage = React.forwardRef<THREE.Group, ImageMessage>(
  function ViserImage(message, ref) {
    const [imageTexture, setImageTexture] = React.useState<THREE.Texture>();

    React.useEffect(() => {
      if (message.props.media_type !== null && message.props._data !== null) {
        const image_url = URL.createObjectURL(new Blob([message.props._data]));
        new THREE.TextureLoader().load(image_url, (texture) => {
          setImageTexture(texture);
          URL.revokeObjectURL(image_url);
        });
      }
    }, [message.props.media_type, message.props._data]);
    return (
      <group ref={ref}>
        <mesh rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}>
          <OutlinesIfHovered />
          <planeGeometry
            attach="geometry"
            args={[message.props.render_width, message.props.render_height]}
          />
          <meshBasicMaterial
            attach="material"
            transparent={true}
            side={THREE.DoubleSide}
            map={imageTexture}
            toneMapped={false}
          />
        </mesh>
      </group>
    );
  },
);

/** Helper for visualizing camera frustums. */
export const CameraFrustum = React.forwardRef<
  THREE.Group,
  {
    fov: number;
    aspect: number;
    scale: number;
    lineWidth: number;
    color: number;
    imageBinary: Uint8Array | null;
    imageMediaType: string | null;
  }
>(function CameraFrustum(props, ref) {
  const [imageTexture, setImageTexture] = React.useState<THREE.Texture>();

  React.useEffect(() => {
    if (props.imageMediaType !== null && props.imageBinary !== null) {
      const image_url = URL.createObjectURL(new Blob([props.imageBinary]));
      new THREE.TextureLoader().load(image_url, (texture) => {
        setImageTexture(texture);
        URL.revokeObjectURL(image_url);
      });
    } else {
      setImageTexture(undefined);
    }
  }, [props.imageMediaType, props.imageBinary]);

  let y = Math.tan(props.fov / 2.0);
  let x = y * props.aspect;
  let z = 1.0;

  const volumeScale = Math.cbrt((x * y * z) / 3.0);
  x /= volumeScale;
  y /= volumeScale;
  z /= volumeScale;
  x *= props.scale;
  y *= props.scale;
  z *= props.scale;

  const hoveredRef = React.useContext(HoverableContext);
  const [isHovered, setIsHovered] = React.useState(false);

  useFrame(() => {
    if (hoveredRef !== null && hoveredRef.current !== isHovered) {
      setIsHovered(hoveredRef.current);
    }
  });

  const frustumPoints: [number, number, number][] = [
    // Rectangle.
    [-1, -1, 1],
    [1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, 1, 1],
    [-1, -1, 1],
    // Lines to origin.
    [-1, -1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [1, -1, 1],
    // Lines to origin.
    [-1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    // Up direction indicator.
    // Don't overlap with the image if the image is present.
    [0.0, -1.2, 1.0],
    imageTexture === undefined ? [0.0, -0.9, 1.0] : [0.0, -1.0, 1.0],
  ].map((xyz) => [xyz[0] * x, xyz[1] * y, xyz[2] * z]);

  return (
    <group ref={ref}>
      <Line
        points={frustumPoints}
        color={isHovered ? 0xfbff00 : props.color}
        lineWidth={isHovered ? 1.5 * props.lineWidth : props.lineWidth}
        segments
      />
      {imageTexture && (
        <mesh
          // 0.999999 is to avoid z-fighting with the frustum lines.
          position={[0.0, 0.0, z * 0.999999]}
          rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
        >
          <planeGeometry
            attach="geometry"
            args={[props.aspect * y * 2, y * 2]}
          />
          <meshBasicMaterial
            attach="material"
            transparent={true}
            side={THREE.DoubleSide}
            map={imageTexture}
            toneMapped={false}
          />
        </mesh>
      )}
    </group>
  );
});

export const HoverableContext =
  React.createContext<React.MutableRefObject<boolean> | null>(null);

/** Outlines object, which should be placed as a child of all meshes that might
 * be clickable. */
export function OutlinesIfHovered(
  props: { alwaysMounted?: boolean; creaseAngle?: number } = {
    // Can be set to true for objects like meshes which may be slow to mount.
    // It seems better to set to False for instanced meshes, there may be some
    // drei or fiber-related race conditions...
    alwaysMounted: false,
    // Some thing just look better with no creasing, like camera frustum objects.
    creaseAngle: Math.PI,
  },
) {
  const groupRef = React.useRef<THREE.Group>(null);
  const hoveredRef = React.useContext(HoverableContext);
  const [mounted, setMounted] = React.useState(true);

  useFrame(() => {
    if (hoveredRef === null) return;
    if (props.alwaysMounted) {
      if (groupRef.current === null) return;
      groupRef.current.visible = hoveredRef.current;
    } else if (hoveredRef.current != mounted) {
      setMounted(hoveredRef.current);
    }
  });
  return hoveredRef === null || !mounted ? null : (
    <Outlines
      ref={groupRef}
      thickness={10}
      screenspace={true}
      color={0xfbff00}
      opacity={0.8}
      transparent={true}
      angle={props.creaseAngle}
    />
  );
}
