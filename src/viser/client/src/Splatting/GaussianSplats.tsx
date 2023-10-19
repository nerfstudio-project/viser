import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { fragmentShaderSource, vertexShaderSource } from "./Shaders";
import { useFrame } from "@react-three/fiber";
import { GaussianBuffersSplitCov } from "./SplatSortWorker";

export type GaussianBuffers = {
  // (N, 3)
  centers: Float32Array;
  // (N, 3)
  rgbs: Float32Array;
  // (N, 1)
  opacities: Float32Array;
  // (N, 6)
  covariancesTriu: Float32Array;
};

export default function GaussianSplats({
  buffers,
}: {
  buffers: GaussianBuffers;
}) {
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [material, setMaterial] = React.useState<THREE.RawShaderMaterial>();
  const setSortedBuffers = React.useRef<
    null | ((sortedBuffers: GaussianBuffersSplitCov) => void)
  >(null);
  const splatSortWorkerRef = React.useRef<Worker | null>(null);

  // We'll use the vanilla three.js API, which for our use case is more
  // flexible than the declarative version (particularly for operations like
  // dynamic updates to buffers and shader uniforms).
  React.useEffect(() => {
    // Create geometry. Each Gaussian will be rendered as a quad.
    const geometry = new THREE.InstancedBufferGeometry();
    const numGaussians = buffers.centers.length / 3;
    geometry.instanceCount = numGaussians;
    geometry.setIndex(
      new THREE.BufferAttribute(new Uint32Array([0, 2, 1, 0, 3, 2]), 1),
    );
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]),
        2,
      ),
    );
    const centerAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(numGaussians * 3),
      3,
    );
    const rgbAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(numGaussians * 3),
      3,
    );
    const opacityAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(numGaussians),
      1,
    );
    const covAAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(numGaussians * 3),
      3,
    );
    const covBAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(numGaussians * 3),
      3,
    );
    geometry.setAttribute("center", centerAttribute);
    geometry.setAttribute("rgb", rgbAttribute);
    geometry.setAttribute("opacity", opacityAttribute);
    geometry.setAttribute("covA", covAAttribute);
    geometry.setAttribute("covB", covBAttribute);

    // Populate material with custom shaders.
    const material = new THREE.RawShaderMaterial({
      fragmentShader: fragmentShaderSource,
      vertexShader: vertexShaderSource,
      uniforms: {
        viewport: { value: null },
        focal: { value: null },
      },
      depthTest: true,
      depthWrite: false,
      transparent: true,
    });

    // Update component state.
    setGeometry(geometry);
    setMaterial(material);
    setSortedBuffers.current = (sortedBuffers: GaussianBuffersSplitCov) => {
      centerAttribute.set(sortedBuffers.centers);
      rgbAttribute.set(sortedBuffers.rgbs);
      opacityAttribute.set(sortedBuffers.opacities);
      covAAttribute.set(sortedBuffers.covA);
      covBAttribute.set(sortedBuffers.covB);

      centerAttribute.needsUpdate = true;
      rgbAttribute.needsUpdate = true;
      opacityAttribute.needsUpdate = true;
      covAAttribute.needsUpdate = true;
      covBAttribute.needsUpdate = true;
    };

    // Create sorting worker.
    const sortWorker = new SplatSortWorker();
    sortWorker.onmessage = (e) => {
      setSortedBuffers.current !== null &&
        setSortedBuffers.current(e.data as GaussianBuffersSplitCov);
    };
    sortWorker.postMessage({
      setBuffers: splitCovariances(buffers),
    });
    splatSortWorkerRef.current = sortWorker;

    // We should always re-send view projection when buffers are replaced.
    prevViewProj.current = undefined;

    return () => {
      geometry.dispose();
      material.dispose();
      sortWorker.postMessage({ close: true });
    };
  }, [buffers]);

  useFrame((state) => {
    const dpr = state.viewport.dpr;
    const fovY =
      ((state.camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * state.viewport.aspect);
    const fy = (dpr * state.size.height) / (2 * Math.tan(fovY / 2));
    const fx = (dpr * state.size.width) / (2 * Math.tan(fovX / 2));

    if (material === undefined) return;
    material.uniforms.focal.value = [fx, fy];
    material.uniforms.viewport.value = [
      state.size.width * dpr,
      state.size.height * dpr,
    ];
  });

  // Synchronize view projection matrix with sort worker.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const prevViewProj = React.useRef<THREE.Matrix4>();
  useFrame((state) => {
    const mesh = meshRef.current;
    const sortWorker = splatSortWorkerRef.current;
    if (mesh === null || sortWorker === null) return;

    // Compute view projection matrix.
    const viewProj = new THREE.Matrix4()
      .multiply(state.camera.projectionMatrix)
      .multiply(state.camera.matrixWorldInverse)
      .multiply(mesh.matrixWorld);

    // If changed, use projection matrix to sort Gaussians.
    if (
      prevViewProj.current === undefined ||
      !viewProj.equals(prevViewProj.current)
    ) {
      sortWorker.postMessage({ setViewProj: viewProj.elements });
      prevViewProj.current = viewProj;
    }
  });

  return <mesh ref={meshRef} geometry={geometry} material={material} />;
}

/** Split upper-triangular terms (6D) of covariance into pair of 3D terms. This
 * lets us pass vec3 arrays into our shader. */
function splitCovariances(buffers: GaussianBuffers) {
  const covA = new Float32Array(buffers.covariancesTriu.length / 2);
  const covB = new Float32Array(buffers.covariancesTriu.length / 2);
  for (let i = 0; i < covA.length; i++) {
    covA[i] = buffers.covariancesTriu[Math.floor(i / 3) * 6 + (i % 3)];
    covB[i] = buffers.covariancesTriu[Math.floor(i / 3) * 6 + (i % 3) + 3];
  }
  return {
    centers: buffers.centers,
    rgbs: buffers.rgbs,
    opacities: buffers.opacities,
    covA: covA,
    covB: covB,
  };
}
