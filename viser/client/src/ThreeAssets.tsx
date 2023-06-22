import { Instance, Instances } from "@react-three/drei";
import React from "react";
import * as THREE from "three";

const axisGeom = new THREE.CylinderGeometry(1.0, 1.0, 1.0, 16, 1);
const xMaterial = new THREE.MeshBasicMaterial({ color: 0xcc0000 });
const yMaterial = new THREE.MeshBasicMaterial({ color: 0x00cc00 });
const zMaterial = new THREE.MeshBasicMaterial({ color: 0x0000cc });

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
  ref
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
                axes_radius * 2.5
              )
            }
          />
          <mesh
            geometry={axisGeom}
            rotation={new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0)}
            position={[0.5 * axes_length, 0.0, 0.0]}
            scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
            material={xMaterial}
          />
          <mesh
            geometry={axisGeom}
            position={[0.0, 0.5 * axes_length, 0.0]}
            scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
            material={yMaterial}
          />
          <mesh
            geometry={axisGeom}
            rotation={new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)}
            position={[0.0, 0.0, 0.5 * axes_length]}
            scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
            material={zMaterial}
          />
        </>
      )}
    </group>
  );
});

// Camera frustum helper. We jitter to prevent z-fighting for overlapping lines.
const jitter = () => Math.random() * 1e-5;
const frustum_points: [number, number, number][] = [];
frustum_points.push([0, 0, 0]);
frustum_points.push([-1, -1, 1]);
frustum_points.push([1, -1, 1]);
frustum_points.push([0, 0, 0]);
frustum_points.push([-1, 1, 1]);
frustum_points.push([1, 1, 1]);
frustum_points.push([0, 0, 0]);
frustum_points.push([-1 + jitter(), 1 + jitter(), 1 + jitter()]);
frustum_points.push([-1, -1, 1]);
frustum_points.push([1 + jitter(), -1 + jitter(), 1 + jitter()]);
frustum_points.push([1, 1, 1]);

const updir_points: [number, number, number][] = [];
updir_points.push([-0.5, -1.2, 1]);
updir_points.push([0.5, -1.2, 1]);
updir_points.push([0.0, -1.5, 1]);
updir_points.push([-0.5, -1.2, 1]);

/** Helper for visualizing camera frustums.

Note that:
 - This is currently just a pyramid, not a frustum. :-)
 - We currently draw two redundant/overlapping lines. This could be optimized. */
export const CameraFrustum = React.forwardRef<
  THREE.Group,
  {
    fov: number;
    aspect: number;
    scale: number;
    color: number;
  }
>(function CameraFrustum(props, ref) {
  const y = Math.tan(props.fov / 2.0);
  const x = y * props.aspect;
  const z = 1.0;

  function scaledLineNode(points: [number, number, number][]) {
    points = points.map((xyz) => [xyz[0] * x, xyz[1] * y, xyz[2] * z]);
    return [...Array(points.length - 1).keys()].map((i) => (
      <SimpleLine
        key={i}
        radius={0.015 * props.scale}
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
      <Instances limit={20} geometry={lineGeom}>
        <meshBasicMaterial color={props.color} />
        {scaledLineNode(frustum_points)}
        {scaledLineNode(updir_points)}
      </Instances>
    </group>
  );
});

const lineGeom = new THREE.CylinderGeometry(1.0, 1.0, 1.0, 3, 1);
function SimpleLine(props: {
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
    rotationAngle
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
