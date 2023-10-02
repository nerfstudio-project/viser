import React, { useEffect } from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { fragmentShaderSource, vertexShaderSource } from "./Shaders";
import { useFrame } from "@react-three/fiber";

type GaussianSplatsBuffers = {
  centers: Float32Array;
  rgbs: Float32Array;
  opacities: Uint8Array;
  covariancesTriu: Float32Array;
};
type GaussianSplatsSortedBuffers = {
  centers: Float32Array;
  rgbs: Float32Array;
  opacities: Float32Array;
  covA: Float32Array;
  covB: Float32Array;
};

export default function GaussianSplats({
  buffers,
}: {
  buffers: GaussianSplatsBuffers;
}) {
  // Create worker for sorting Gaussians.
  const splatSortWorkerRef = React.useRef<Worker | null>(null);
  const [sortedBuffers, setSortedBuffers] =
    React.useState<null | GaussianSplatsSortedBuffers>(null);
  useEffect(() => {
    const sortWorker = new SplatSortWorker();
    console.log("posting buffers");
    sortWorker.postMessage({ setBuffers: buffers });
    splatSortWorkerRef.current = sortWorker;
    sortWorker.onmessage = (e) => {
      const { centers, rgbs, opacities, covA, covB } = e.data;
      setSortedBuffers({
        centers: centers,
        rgbs: rgbs,
        opacities: opacities,
        covA: covA,
        covB: covB,
      });
    };

    // Close the worker when done.
    return () => sortWorker.postMessage({ close: true });
  }, [buffers]);

  // Synchronize view projection matrix with sort worker.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const shaderRef = React.useRef<THREE.RawShaderMaterial>(null);
  useFrame((state) => {
    const mesh = meshRef.current;
    const shader = shaderRef.current;
    const sortWorker = splatSortWorkerRef.current;
    if (mesh === null || shader === null || sortWorker === null) return;

    // Update shader uniforms.
    const dpr = state.viewport.dpr;
    const fovY =
      ((state.camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * state.viewport.aspect);
    const fy = (dpr * state.size.height) / (2 * Math.tan(fovY / 2));
    const fx = (dpr * state.size.width) / (2 * Math.tan(fovX / 2));

    shader.uniforms.focal.value = [fx, fy];
    shader.uniforms.viewport.value = [
      state.size.width * dpr * (1.0 + Math.random()),
      state.size.height * dpr,
    ];

    // Update sorting.
    const viewProj = new THREE.Matrix4()
      .multiply(state.camera.projectionMatrix)
      .multiply(state.camera.matrixWorldInverse)
      .multiply(mesh.matrixWorld);
    sortWorker.postMessage({ setViewProj: viewProj.elements });
  });

  const numGaussians =
    sortedBuffers === null ? 0 : sortedBuffers.centers.length / 3;
  return (
    <mesh ref={meshRef}>
      {sortedBuffers === null ? null : (
        <instancedBufferGeometry instanceCount={numGaussians}>
          <instancedBufferAttribute
            attach="attributes-center"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={sortedBuffers.centers}
            itemSize={3}
            count={numGaussians}
          />
          <instancedBufferAttribute
            attach="attributes-color"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={sortedBuffers.rgbs}
            itemSize={3}
            count={numGaussians}
          />
          <instancedBufferAttribute
            attach="attributes-opacity"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={sortedBuffers.opacities}
            itemSize={1}
            count={numGaussians}
          />
          <instancedBufferAttribute
            attach="attributes-covA"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={sortedBuffers.covA}
            itemSize={3}
            count={numGaussians}
          />
          <instancedBufferAttribute
            attach="attributes-covB"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={sortedBuffers.covB}
            itemSize={3}
            count={numGaussians}
          />
          <bufferAttribute
            attach="index"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={
              // Each quad should consist of two triangles.
              new Uint32Array([0, 2, 1, 0, 3, 2])
            }
            itemSize={1}
            count={6}
          />
          <bufferAttribute
            attach="attributes-position"
            onUpdate={(self) => {
              self.needsUpdate = true;
            }}
            array={new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2])}
            itemSize={2}
            count={4}
          />
        </instancedBufferGeometry>
      )}
      <rawShaderMaterial
        ref={shaderRef}
        uniforms={{ viewport: { value: null }, focal: { value: null } }}
        fragmentShader={fragmentShaderSource}
        vertexShader={vertexShaderSource}
        depthTest={true}
        depthWrite={false}
        transparent={true}
      />
    </mesh>
  );
}
