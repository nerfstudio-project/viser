import { InstancedMesh2 } from "@three.ez/instanced-mesh";
import { Instance, Instances, Line, shaderMaterial } from "@react-three/drei";
import { createPortal, useFrame, useThree } from "@react-three/fiber";
import { Outlines } from "./Outlines";
import React from "react";
import { HoverableContext } from "./HoverContext";
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
  CameraFrustumMessage,
  ImageMessage,
  MeshMessage,
  PointCloudMessage,
  SkinnedMeshMessage,
  BatchedMeshesMessage,
  GlbMessage,
  BatchedGlbMessage,
} from "./WebsocketMessages";
import { ViewerContext } from "./ViewerContext";
import { MeshoptSimplifier } from "meshoptimizer";

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

function getAutoLODSettings(
  mesh: THREE.Mesh,
  scale: number = 1,
): { ratios: number[]; distances: number[] } {
  // Heuristics for automatic LOD parameters.
  const geometry = mesh.geometry;
  const boundingRadius = geometry.boundingSphere!.radius * scale;
  const vertexCount = geometry.attributes.position.count;

  // 1. Compute LOD ratios, based on vertex count.
  let ratios: number[];
  if (vertexCount > 10_000) {
    ratios = [0.2, 0.05, 0.01]; // very complex
  } else if (vertexCount > 2_000) {
    ratios = [0.4, 0.1, 0.03]; // medium complex
  } else if (vertexCount > 500) {
    ratios = [0.6, 0.2, 0.05]; // light
  } else {
    ratios = [0.85, 0.4, 0.1]; // already simple
  }

  // 2. Compute LOD distances, based on bounding radius.
  const sizeFactor = Math.sqrt(boundingRadius + 1e-5);
  const baseMultipliers = [1, 2, 3]; // distance "steps" for LOD switching
  const distances = baseMultipliers.map((m) => m * sizeFactor);

  return { ratios, distances };
}

// The LOD functionality has been moved into the BatchedMeshManager class

// The helper function has been moved into the BatchedMeshManager class

/**
 * Helper class to manage batched mesh instances and ensure proper resource disposal
 * This class ensures that all resources (geometries, LODs) are properly cleaned up
 */
class BatchedMeshManager {
  private instancedMesh: InstancedMesh2;
  private lodGeometries: THREE.BufferGeometry[] = [];

  constructor(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    numInstances: number,
    lodSetting: "off" | "auto" | [number, number][],
    castShadow: boolean,
    scale?: number,
  ) {
    this.instancedMesh = new InstancedMesh2(geometry, material);

    // Setup LODs if needed
    if (lodSetting !== "off") {
      this.setupLODs(geometry, material, lodSetting, castShadow, scale);
    }

    // Setup instances
    this.instancedMesh.addInstances(numInstances, () => {});
  }

  private setupLODs(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    lodSetting: "auto" | [number, number][],
    castShadow: boolean,
    scale?: number,
  ) {
    const dummyMesh = new THREE.Mesh(geometry, material);

    if (lodSetting === "auto") {
      const { ratios, distances } = getAutoLODSettings(dummyMesh, scale);
      this.addLODs(dummyMesh, ratios, distances, castShadow);
    } else {
      this.addLODs(
        dummyMesh,
        lodSetting.map((pair) => pair[1]),
        lodSetting.map((pair) => pair[0]),
        castShadow,
      );
    }
  }

  private addLODs(
    mesh: THREE.Mesh,
    ratios: number[],
    distances: number[],
    castShadow: boolean,
  ) {
    ratios.forEach((ratio, index) => {
      const targetCount =
        Math.floor((mesh.geometry.index!.array.length * ratio) / 3) * 3;
      const lodGeometry = mesh.geometry.clone();

      const dstIndexArray = MeshoptSimplifier.simplify(
        new Uint32Array(lodGeometry.index!.array),
        new Float32Array(lodGeometry.attributes.position.array),
        3,
        targetCount,
        0.01, // Error tolerance.
        ["LockBorder"], // Important to avoid triangle flipping artifacts.
      )[0];

      lodGeometry.index!.array.set(dstIndexArray);
      lodGeometry.index!.needsUpdate = true;
      lodGeometry.setDrawRange(0, dstIndexArray.length);
      this.instancedMesh.addLOD(lodGeometry, mesh.material, distances[index]);

      if (castShadow) {
        this.instancedMesh.addShadowLOD(lodGeometry, distances[index]);
      }

      // Store the geometry for proper disposal later
      this.lodGeometries.push(lodGeometry);
    });
  }

  /** Update instance transforms */
  updateInstances(
    batched_positions: Float32Array,
    batched_wxyzs: Float32Array,
    meshTransform?: {
      position: THREE.Vector3;
      rotation: THREE.Quaternion;
      scale: THREE.Vector3;
    },
  ) {
    this.instancedMesh.updateInstances((obj, index) => {
      // Create instance world transform
      const instanceWorldMatrix = new THREE.Matrix4().compose(
        new THREE.Vector3(
          batched_positions[index * 3 + 0],
          batched_positions[index * 3 + 1],
          batched_positions[index * 3 + 2],
        ),
        new THREE.Quaternion(
          batched_wxyzs[index * 4 + 1],
          batched_wxyzs[index * 4 + 2],
          batched_wxyzs[index * 4 + 3],
          batched_wxyzs[index * 4 + 0],
        ),
        new THREE.Vector3(1, 1, 1),
      );

      if (meshTransform) {
        // Apply mesh's original transform relative to the instance
        const meshMatrix = new THREE.Matrix4().compose(
          meshTransform.position,
          meshTransform.rotation,
          new THREE.Vector3(1, 1, 1),
        );

        // Combine transforms and apply
        const finalMatrix = instanceWorldMatrix.multiply(meshMatrix);
        obj.position.setFromMatrixPosition(finalMatrix);
        obj.quaternion.setFromRotationMatrix(finalMatrix);
        obj.scale.copy(meshTransform.scale);
      } else {
        // Direct instance transform without mesh offset
        obj.position.setFromMatrixPosition(instanceWorldMatrix);
        obj.quaternion.setFromRotationMatrix(instanceWorldMatrix);
      }
    });
  }

  /** Update the number of instances */
  setInstanceCount(numInstances: number) {
    this.instancedMesh.clearInstances();
    this.instancedMesh.addInstances(numInstances, () => {});
  }

  /** Get the instanced mesh object */
  getMesh() {
    return this.instancedMesh;
  }

  /** Dispose all resources */
  dispose() {
    // Dispose all LOD geometries
    for (const geometry of this.lodGeometries) {
      geometry.dispose();
    }

    // The instancedMesh will dispose its main geometry and material
    this.instancedMesh.dispose();
  }
}

/** Helper function to setup batched mesh instances */
function setupBatchedMesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  numInstances: number,
  lodSetting: "off" | "auto" | [number, number][],
  castShadow: boolean,
  scale?: number,
): BatchedMeshManager {
  return new BatchedMeshManager(
    geometry,
    material,
    numInstances,
    lodSetting,
    castShadow,
    scale,
  );
}

const PointCloudMaterial = /* @__PURE__ */ shaderMaterial(
  {
    scale: 1.0,
    point_ball_norm: 0.0,
    uniformColor: new THREE.Color(1, 1, 1),
  },
  `
  precision mediump float;

  varying vec3 vPosition;
  varying vec3 vColor; // in the vertex shader
  uniform float scale;
  uniform vec3 uniformColor;

  void main() {
      vPosition = position;
      #ifdef USE_COLOR
      vColor = color;
      #else
      vColor = uniformColor;
      #endif
      vec4 world_pos = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * world_pos;
      gl_PointSize = (scale / -world_pos.z);
  }
   `,
  `varying vec3 vPosition;
  varying vec3 vColor;
  uniform float point_ball_norm;
  uniform vec3 uniformColor;

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

export const PointCloud = React.forwardRef<THREE.Points, PointCloudMessage>(
  function PointCloud(message, ref) {
    const getThreeState = useThree((state) => state.get);

    const props = message.props;

    const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
    const [material, setMaterial] = React.useState<ShaderMaterial>();

    React.useEffect(() => {
      const geometry = new THREE.BufferGeometry();

      geometry.setAttribute(
        "position",
        new THREE.Float16BufferAttribute(
          new Uint16Array(
            props.points.buffer.slice(
              props.points.byteOffset,
              props.points.byteOffset + props.points.byteLength,
            ),
          ),
          3,
        ),
      );

      const material = new PointCloudMaterial();
      if (props.colors.length > 3) {
        material.vertexColors = true;
        geometry.setAttribute(
          "color",
          new THREE.BufferAttribute(new Uint8Array(props.colors), 3, true),
        );
      } else {
        if (props.colors.length < 3) {
          console.error(
            `Invalid color buffer length, got ${props.colors.length}`,
          );
        }
        material.vertexColors = false;
        material.uniforms.uniformColor.value = new THREE.Color(
          props.colors[0],
          props.colors[1],
          props.colors[2],
        );
      }
      setGeometry(geometry);
      setMaterial(material);
    }, [props.points, props.colors]);

    React.useEffect(() => {
      return () => {
        if (geometry === undefined) return;
        geometry.dispose();
      };
    }, [geometry]);

    React.useEffect(() => {
      return () => {
        if (material === undefined) return;
        material.dispose();
      };
    }, [material]);

    React.useEffect(() => {
      if (material === undefined) return;
      material.uniforms.scale.value = 10.0;
      material.uniforms.point_ball_norm.value = props.point_ball_norm;
    }, [props.point_ball_norm, material]);

    const rendererSize = new THREE.Vector2();
    useFrame(() => {
      // Match point scale to behavior of THREE.PointsMaterial().
      if (material === undefined) return;
      // point px height / actual height = point meters height / frustum meters height
      // frustum meters height = math.tan(fov / 2.0) * z
      // point px height = (point meters height / math.tan(fov / 2.0) * actual height)  / z
      material.uniforms.scale.value =
        (props.point_size /
          Math.tan(
            (((getThreeState().camera as THREE.PerspectiveCamera).fov / 180.0) *
              Math.PI) /
              2.0,
          )) *
        getThreeState().gl.getSize(rendererSize).height *
        getThreeState().gl.getPixelRatio();
    });
    return <points ref={ref} geometry={geometry} material={material} />;
  },
);

/** Component for rendering the contents of GLB files. */
export const GlbAsset = React.forwardRef<
  THREE.Group,
  GlbMessage | BatchedGlbMessage
>(function GlbAsset(message, ref) {
  const props = message.props;

  // Create persistent geometry and material. Set attributes when we receive updates.
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

    const glb_data = new Uint8Array(props.glb_data);
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
          if (obj instanceof THREE.Mesh) {
            obj.geometry.computeVertexNormals();
            obj.geometry.computeBoundingSphere();
            obj.castShadow = props.cast_shadow;
            obj.receiveShadow = props.receive_shadow;
            meshes.push(obj);
          }
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
  }, [props.glb_data, props.cast_shadow, props.receive_shadow]);

  useFrame((_, delta) => {
    if (mixerRef.current) {
      mixerRef.current.update(delta);
    }
  });

  // Create the instanced meshes for batched GLBs:
  // - Flattening the hierarchy to store root-to-node transformations, then
  // - Applying both the root-to-world and node-to-root transformations.
  // We additionally generate LODs for each mesh, if requested.
  const [transforms, setTransforms] = React.useState<
    {
      position: THREE.Vector3;
      rotation: THREE.Quaternion;
      scale: THREE.Vector3;
    }[]
  >([]);

  // Use a ref to store mesh managers for proper disposal
  const meshManagersRef = React.useRef<{
    gltfScene: THREE.Group;
    managers: BatchedMeshManager[];
  } | null>(null);

  // Initialize mesh managers
  React.useEffect(() => {
    if (message.type !== "BatchedGlbMessage" || !gltf) return;

    // Clean up previous managers if they exist
    if (meshManagersRef.current) {
      meshManagersRef.current.managers.forEach((manager) => {
        manager.dispose();
      });
    }

    const scene = gltf.scene.clone();
    const managers: BatchedMeshManager[] = [];
    const transforms: {
      position: THREE.Vector3;
      rotation: THREE.Quaternion;
      scale: THREE.Vector3;
    }[] = [];

    scene.traverse((node) => {
      if (node instanceof THREE.Mesh && node.parent) {
        // Store transform info.
        const position = new THREE.Vector3();
        const scale = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        node.getWorldPosition(position);
        node.getWorldScale(scale);
        node.getWorldQuaternion(quat);

        const transform = {
          position: position.clone(),
          rotation: quat.clone(),
          scale: scale.clone(),
        };
        transforms.push(transform);

        // Create instanced mesh with LOD
        const numInstances =
          message.props.batched_positions.byteLength /
          (3 * Float32Array.BYTES_PER_ELEMENT);
        const manager = setupBatchedMesh(
          node.geometry.clone(),
          node.material,
          numInstances,
          message.props.lod,
          message.props.cast_shadow,
          Math.max(scale.x, scale.y, scale.z),
        );

        // Hide the original node.
        node.visible = false;

        // Add the instanced mesh to the scene.
        managers.push(manager);
        scene.add(manager.getMesh());
      }
    });

    setTransforms(transforms);
    meshManagersRef.current = { gltfScene: scene, managers };

    // Clean up when component unmounts or dependencies change
    return () => {
      if (meshManagersRef.current) {
        meshManagersRef.current.managers.forEach((manager) => {
          manager.dispose();
        });
        meshManagersRef.current = null;
      }
    };
  }, [
    message.type,
    gltf,
    ...(message.type === "BatchedGlbMessage"
      ? [message.props.lod, message.props.cast_shadow]
      : []),
  ]);

  // Handle updates to instance positions/orientations.
  React.useEffect(() => {
    if (message.type !== "BatchedGlbMessage" || !meshManagersRef.current)
      return;

    const batched_positions = new Float32Array(
      message.props.batched_positions.buffer.slice(
        message.props.batched_positions.byteOffset,
        message.props.batched_positions.byteOffset +
          message.props.batched_positions.byteLength,
      ),
    );

    const batched_wxyzs = new Float32Array(
      message.props.batched_wxyzs.buffer.slice(
        message.props.batched_wxyzs.byteOffset,
        message.props.batched_wxyzs.byteOffset +
          message.props.batched_wxyzs.byteLength,
      ),
    );

    // Update instance count if needed.
    const newNumInstances =
      message.props.batched_positions.byteLength /
      (3 * Float32Array.BYTES_PER_ELEMENT);

    // Update all mesh managers
    meshManagersRef.current.managers.forEach((manager, mesh_index) => {
      manager.setInstanceCount(newNumInstances);
      manager.updateInstances(
        batched_positions,
        batched_wxyzs,
        transforms[mesh_index],
      );
    });
  }, [
    message.type,
    transforms,
    ...(message.type === "BatchedGlbMessage"
      ? [message.props.batched_positions, message.props.batched_wxyzs]
      : []),
  ]);

  if (!gltf) return null;

  return (
    <group ref={ref}>
      {message.type === "BatchedGlbMessage" ? (
        meshManagersRef.current && (
          <>
            <primitive
              object={meshManagersRef.current.gltfScene}
              scale={message.props.scale}
            />
            <OutlinesIfHovered alwaysMounted={false} />
          </>
        )
      ) : (
        <>
          <primitive
            object={gltf.scene}
            scale={message.props.scale}
            castShadow={message.props.cast_shadow}
            receiveShadow={message.props.receive_shadow}
          />
          {meshes.map((mesh, i) => (
            <React.Fragment key={i}>
              {createPortal(<OutlinesIfHovered alwaysMounted />, mesh)}
            </React.Fragment>
          ))}
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
    batched_wxyzs: Float32Array;
    batched_positions: Float32Array;
    axes_length?: number;
    axes_radius?: number;
  }
>(function InstancedAxes(
  { batched_wxyzs, batched_positions, axes_length = 0.5, axes_radius = 0.0125 },
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

    for (let i = 0; i < batched_wxyzs.length / 4; i++) {
      T_world_frame.makeRotationFromQuaternion(
        tmpQuat.set(
          batched_wxyzs[i * 4 + 1],
          batched_wxyzs[i * 4 + 2],
          batched_wxyzs[i * 4 + 3],
          batched_wxyzs[i * 4 + 0],
        ),
      ).setPosition(
        batched_positions[i * 3 + 0],
        batched_positions[i * 3 + 1],
        batched_positions[i * 3 + 2],
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
  }, [batched_wxyzs, batched_positions]);

  return (
    <group ref={ref}>
      <instancedMesh
        ref={axesRef}
        args={[cylinderGeom, material, (batched_wxyzs.length / 4) * 3]}
      >
        <OutlinesIfHovered />
      </instancedMesh>
    </group>
  );
});

/** Convert raw RGB color buffers to linear color buffers. **/
export const ViserMesh = React.forwardRef<
  THREE.Mesh | THREE.SkinnedMesh | InstancedMesh2,
  MeshMessage | SkinnedMeshMessage | BatchedMeshesMessage
>(function ViserMesh(message, ref) {
  const viewer = React.useContext(ViewerContext)!;

  const generateGradientMap = (shades: 3 | 5) => {
    const texture = new THREE.DataTexture(
      Uint8Array.from(shades == 3 ? [0, 128, 255] : [0, 64, 128, 192, 255]),
      shades,
      1,
      THREE.RedFormat,
    );

    texture.needsUpdate = true;
    return texture;
  };

  const assertUnreachable = (x: never): never => {
    throw new Error(`Should never get here! ${x}`);
  };

  const [material, setMaterial] = React.useState<THREE.Material>();
  const bonesRef = React.useRef<THREE.Bone[]>();

  React.useEffect(() => {
    const standardArgs = {
      color:
        message.props.color === null
          ? undefined
          : rgbToInt(message.props.color),
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
    setMaterial(material);

    return () => {
      // Dispose material when done.
      material.dispose();
    };
  }, [
    message.props.material,
    message.props.color,
    message.props.wireframe,
    message.props.opacity,
  ]);

  // Create persistent geometry. Set attributes when we receive updates.
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [skeleton, setSkeleton] = React.useState<THREE.Skeleton>();

  // Setup geometry attributes
  React.useEffect(() => {
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

    // Handle skinned mesh setup if needed
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

        bone.setRotationFromQuaternion(xyzw_quat);
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
    skeleton?.init();
    setGeometry(geometry);
    setSkeleton(skeleton);
    return () => {
      if (message.type === "SkinnedMeshMessage") {
        skeleton !== undefined && skeleton.dispose();
        const state = viewer.skinnedMeshState.current[message.name];
        state.initialized = false;
      }
    };
  }, [
    message.type,
    message.props.vertices.buffer,
    message.props.faces.buffer,
    message.type == "SkinnedMeshMessage"
      ? message.props.skin_indices.buffer
      : null,
    message.type == "SkinnedMeshMessage"
      ? message.props.skin_weights.buffer
      : null,
    message.type == "SkinnedMeshMessage"
      ? message.props.bone_wxyzs.buffer
      : null,
    message.type == "SkinnedMeshMessage"
      ? message.props.bone_positions.buffer
      : null,
  ]);
  // Dispose geometry when done.
  React.useEffect(() => {
    return () => {
      geometry !== undefined && geometry.dispose();
    };
  }, [geometry]);

  // Update bone transforms for skinned meshes.
  useFrame(() => {
    if (message.type !== "SkinnedMeshMessage") return;

    const parentNode = viewer.nodeRefFromName.current[message.name];
    if (parentNode === undefined) return;

    const state = viewer.skinnedMeshState.current[message.name];
    const bones = bonesRef.current;
    if (skeleton !== undefined && bones !== undefined) {
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

  // Create the batched mesh manager once
  const meshManagerRef = React.useRef<BatchedMeshManager | null>(null);

  // Create the instanced mesh once
  React.useEffect(() => {
    if (
      message.type !== "BatchedMeshesMessage" ||
      material === undefined ||
      geometry === undefined
    )
      return;

    const numInstances =
      message.props.batched_positions.byteLength /
      (3 * Float32Array.BYTES_PER_ELEMENT);

    // Create new manager
    meshManagerRef.current = setupBatchedMesh(
      geometry,
      material,
      numInstances,
      message.props.lod,
      message.props.cast_shadow,
    );

    // Cleanup function to ensure proper disposal
    return () => {
      if (meshManagerRef.current) {
        meshManagerRef.current.dispose();
        meshManagerRef.current = null;
      }
    };
  }, [
    message.type,
    material,
    geometry,
    ...(message.type === "BatchedMeshesMessage"
      ? [message.props.lod, message.props.cast_shadow]
      : []),
  ]);

  // Handle updates to instance positions/orientations
  React.useEffect(() => {
    if (message.type !== "BatchedMeshesMessage" || !meshManagerRef.current)
      return;

    const batched_positions = new Float32Array(
      message.props.batched_positions.buffer.slice(
        message.props.batched_positions.byteOffset,
        message.props.batched_positions.byteOffset +
          message.props.batched_positions.byteLength,
      ),
    );

    const batched_wxyzs = new Float32Array(
      message.props.batched_wxyzs.buffer.slice(
        message.props.batched_wxyzs.byteOffset,
        message.props.batched_wxyzs.byteOffset +
          message.props.batched_wxyzs.byteLength,
      ),
    );

    // Update instance count if needed
    const newNumInstances =
      message.props.batched_positions.byteLength /
      (3 * Float32Array.BYTES_PER_ELEMENT);
    meshManagerRef.current.setInstanceCount(newNumInstances);

    // Update instance transforms
    meshManagerRef.current.updateInstances(batched_positions, batched_wxyzs);
  }, [
    message.type,
    ...(message.type === "BatchedMeshesMessage"
      ? [
          message.props.batched_positions.buffer,
          message.props.batched_wxyzs.buffer,
        ]
      : []),
  ]);

  if (geometry === undefined || material === undefined) {
    return null;
  }

  // Render the appropriate mesh type
  if (message.type === "BatchedMeshesMessage") {
    return (
      <>
        {meshManagerRef.current && (
          <>
            <primitive object={meshManagerRef.current.getMesh()} />
            <OutlinesIfHovered alwaysMounted />
          </>
        )}
      </>
    );
  } else if (message.type === "SkinnedMeshMessage") {
    return (
      <skinnedMesh
        ref={ref as React.ForwardedRef<THREE.SkinnedMesh>}
        geometry={geometry}
        material={material}
        skeleton={skeleton}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
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
    return (
      <mesh
        ref={ref as React.ForwardedRef<THREE.Mesh>}
        geometry={geometry}
        material={material}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
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
        <mesh
          rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
          castShadow={message.props.cast_shadow}
          receiveShadow={message.props.receive_shadow}
        >
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
  CameraFrustumMessage
>(function CameraFrustum(message, ref) {
  const [imageTexture, setImageTexture] = React.useState<THREE.Texture>();

  React.useEffect(() => {
    if (
      message.props.image_media_type !== null &&
      message.props._image_data !== null
    ) {
      const image_url = URL.createObjectURL(
        new Blob([message.props._image_data]),
      );
      new THREE.TextureLoader().load(image_url, (texture) => {
        setImageTexture(texture);
        URL.revokeObjectURL(image_url);
      });
    } else {
      setImageTexture(undefined);
    }
  }, [message.props.image_media_type, message.props._image_data]);

  let y = Math.tan(message.props.fov / 2.0);
  let x = y * message.props.aspect;
  let z = 1.0;

  const volumeScale = Math.cbrt((x * y * z) / 3.0);
  x /= volumeScale;
  y /= volumeScale;
  z /= volumeScale;
  x *= message.props.scale;
  y *= message.props.scale;
  z *= message.props.scale;

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
        color={isHovered ? 0xfbff00 : rgbToInt(message.props.color)}
        lineWidth={
          isHovered ? 1.5 * message.props.line_width : message.props.line_width
        }
        segments
      />
      {imageTexture && (
        <mesh
          // 0.999999 is to avoid z-fighting with the frustum lines.
          position={[0.0, 0.0, z * 0.999999]}
          rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
          castShadow={message.props.cast_shadow}
          receiveShadow={message.props.receive_shadow}
        >
          <planeGeometry
            attach="geometry"
            args={[message.props.aspect * y * 2, y * 2]}
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
