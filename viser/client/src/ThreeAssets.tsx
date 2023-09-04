import { Instance, Instances } from "@react-three/drei";
import React from "react";
import * as THREE from "three";

const axisGeom = new THREE.CylinderGeometry(1.0, 1.0, 1.0, 16, 1);
const originGeom = new THREE.SphereGeometry(1.0);
const originMaterial = new THREE.MeshBasicMaterial({ color: 0xecec00 });

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
          />
          <Instances geometry={axisGeom}>
            <meshBasicMaterial />
            <Instance
              rotation={new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0)}
              position={[0.5 * axes_length, 0.0, 0.0]}
              scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
              color={0xcc0000}
            />
            <Instance
              position={[0.0, 0.5 * axes_length, 0.0]}
              scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
              color={0x00cc00}
            />
            <Instance
              rotation={new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)}
              position={[0.0, 0.0, 0.5 * axes_length]}
              scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
              color={0x0000cc}
            />
          </Instances>
        </>
      )}
    </group>
  );
});

const lineGeom = new THREE.CylinderGeometry(1.0, 1.0, 1.0, 3, 1);
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
        radius={0.06 * props.scale}
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
      <Instances limit={9} geometry={lineGeom}>
        <meshBasicMaterial color={props.color} />
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
  radius: number;
  color: number;
}) {
  const desiredDirection = new THREE.Vector3()
    .subVectors(props.end, props.start)
    .normalize();
  const canonicalDirection = new THREE.Vector3(0.0, 1.0, 0.0);
  const rotationAxis = new THREE.Vector3()
    .copy(canonicalDirection)
    .cross(desiredDirection)
    .normalize();
  const rotationAngle = Math.acos(desiredDirection.dot(canonicalDirection));

  const length = props.start.distanceTo(props.end);
  const midpoint = new THREE.Vector3()
    .addVectors(props.start, props.end)
    .divideScalar(2.0);

  const orientation = new THREE.Quaternion().setFromAxisAngle(
    rotationAxis,
    rotationAngle,
  );
  return (
    <>
      <Instance
        position={midpoint}
        quaternion={orientation}
        scale={[props.radius, length, props.radius]}
      />
    </>
  );
}
