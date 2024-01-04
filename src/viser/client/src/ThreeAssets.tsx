import { Instance, Instances } from "@react-three/drei";
import { createPortal, useFrame } from "@react-three/fiber";
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

const originGeom = new THREE.SphereGeometry(1.0);
const originMaterial = new THREE.MeshBasicMaterial({ color: 0xecec00 });

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
    show_axes?: boolean;
    axes_length?: number;
    axes_radius?: number;
  }
>(function CoordinateFrame(
  { show_axes = true, axes_length = 0.5, axes_radius = 0.0125 },
  ref,
) {
  return (
    <group ref={ref}>
      {show_axes && (
        <>
          <mesh
            geometry={originGeom}
            material={originMaterial}
            scale={
              new THREE.Vector3(
                axes_radius * 2.5,
                axes_radius * 2.5,
                axes_radius * 2.5,
              )
            }
          >
            <OutlinesIfHovered />
          </mesh>
          <Instances>
            <meshBasicMaterial />
            <cylinderGeometry
              args={[axes_radius, axes_radius, axes_length, 16]}
            />
            <Instance
              rotation={new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0)}
              position={[0.5 * axes_length, 0.0, 0.0]}
              color={0xcc0000}
            >
              <OutlinesIfHovered />
            </Instance>
            <Instance position={[0.0, 0.5 * axes_length, 0.0]} color={0x00cc00}>
              <OutlinesIfHovered />
            </Instance>
            <Instance
              rotation={new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)}
              position={[0.0, 0.0, 0.5 * axes_length]}
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

/** Helper for visualizing camera frustums. */
export const CameraFrustum = React.forwardRef<
  THREE.Group,
  {
    fov: number;
    aspect: number;
    scale: number;
    color: number;
    image?: THREE.Texture;
  }
>(function CameraFrustum(props, ref) {
  let y = Math.tan(props.fov / 2.0);
  let x = y * props.aspect;
  let z = 1.0;

  const volumeScale = Math.cbrt((x * y * z) / 3.0);
  x /= volumeScale;
  y /= volumeScale;
  z /= volumeScale;

  function scaledLineSegments(points: [number, number, number][]) {
    points = points.map((xyz) => [xyz[0] * x, xyz[1] * y, xyz[2] * z]);
    return [...Array(points.length - 1).keys()].map((i) => (
      <LineSegmentInstance
        key={i}
        start={new THREE.Vector3()
          .fromArray(points[i])
          .multiplyScalar(props.scale)}
        end={new THREE.Vector3()
          .fromArray(points[i + 1])
          .multiplyScalar(props.scale)}
        color={props.color}
      />
    ));
  }

  return (
    <group ref={ref}>
      <Instances limit={9}>
        <meshBasicMaterial color={props.color} side={THREE.DoubleSide} />
        <cylinderGeometry
          args={[props.scale * 0.03, props.scale * 0.03, 1.0, 3]}
        />
        {scaledLineSegments([
          // Rectangle.
          [-1, -1, 1],
          [1, -1, 1],
          [1, 1, 1],
          [-1, 1, 1],
          [-1, -1, 1],
        ])}
        {scaledLineSegments([
          // Lines to origin.
          [-1, -1, 1],
          [0, 0, 0],
          [1, -1, 1],
        ])}
        {scaledLineSegments([
          // Lines to origin.
          [-1, 1, 1],
          [0, 0, 0],
          [1, 1, 1],
        ])}
        {scaledLineSegments([
          // Up direction.
          [0.0, -1.2, 1.0],
          [0.0, -0.9, 1.0],
        ])}
      </Instances>
      {props.image && (
        <mesh
          position={[0.0, 0.0, props.scale * z]}
          rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
        >
          <planeGeometry
            attach="geometry"
            args={[props.scale * props.aspect * y * 2, props.scale * y * 2]}
          />
          <meshBasicMaterial
            attach="material"
            transparent={true}
            side={THREE.DoubleSide}
            map={props.image}
          />
        </mesh>
      )}
    </group>
  );
});

function LineSegmentInstance(props: {
  start: THREE.Vector3;
  end: THREE.Vector3;
  color: number;
}) {
  const desiredDirection = new THREE.Vector3()
    .subVectors(props.end, props.start)
    .normalize();
  const canonicalDirection = new THREE.Vector3(0.0, 1.0, 0.0);
  const orientation = new THREE.Quaternion().setFromUnitVectors(
    canonicalDirection,
    desiredDirection,
  );

  const length = props.start.distanceTo(props.end);
  const midpoint = new THREE.Vector3()
    .addVectors(props.start, props.end)
    .divideScalar(2.0);

  return (
    <Instance
      position={midpoint}
      quaternion={orientation}
      scale={[1.0, length, 1.0]}
    >
      <OutlinesIfHovered creaseAngle={0.0} />
    </Instance>
  );
}

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
