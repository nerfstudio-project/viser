import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame } from "@react-three/fiber";
import { GaussianBuffersSplitCov } from "./SplatSortWorker";
import { shaderMaterial } from "@react-three/drei";

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

const GaussianSplatMaterial = /* @__PURE__ */ shaderMaterial(
  {
    alphaTest: 0,
    alphaHash: false,
    viewport: [1920, 1080],
    focal: 100.0,
    depthTest: true,
    depthWrite: false,
    transparent: true,
  },
  `precision mediump float;

  attribute vec3 rgb;
  attribute float opacity;
  attribute vec3 center;
  attribute vec3 covA;
  attribute vec3 covB;

  uniform vec2 focal;
  uniform vec2 viewport;

  varying vec3 vRgb;
  varying float vOpacity;
  varying vec2 vPosition;

  void main () {
    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec4 c_cam = modelViewMatrix * vec4(center, 1);
    vec4 pos2d = projectionMatrix * c_cam;

    // Splat covariance.
    mat3 cov3d = mat3(
        covA.x, covA.y, covA.z,
        covA.y, covB.x, covB.y,
        covA.z, covB.y, covB.z
    );
    mat3 J = mat3(
        // Matrices are column-major.
        focal.x / c_cam.z, 0., 0.0,
        0., focal.y / c_cam.z, 0.0,
        -(focal.x * c_cam.x) / (c_cam.z * c_cam.z), -(focal.y * c_cam.y) / (c_cam.z * c_cam.z), 0.
    );
    mat3 A = J * mat3(modelViewMatrix);
    mat3 cov_proj = A * cov3d * transpose(A);
    float diag1 = cov_proj[0][0] + 0.3;
    float offDiag = cov_proj[0][1];
    float diag2 = cov_proj[1][1] + 0.3;

    // Eigendecomposition. This can mostly be derived from characteristic equation, etc.
    float mid = 0.5 * (diag1 + diag2);
    float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    vec2 diagonalVector = normalize(vec2(offDiag, lambda1 - diag1));
    vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vRgb = rgb;
    vOpacity = opacity;
    vPosition = vec2(position.x, position.y);

    gl_Position = vec4(
        vec2(pos2d) / pos2d.w
            + position.x * v1 / viewport * 2.0
            + position.y * v2 / viewport * 2.0, pos2d.z / pos2d.w, 1.);
  }
`,
  `#include <alphatest_pars_fragment>
  #include <alphahash_pars_fragment>

  precision mediump float;

  varying vec3 vRgb;
  varying float vOpacity;
  varying vec2 vPosition;

  uniform vec2 viewport;
  uniform vec2 focal;


  void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vOpacity;
    vec4 diffuseColor = vec4(vRgb.rgb, B);
    #include <alphatest_fragment>
    #include <alphahash_fragment>
    gl_FragColor = diffuseColor;
  }`,
);

export default function GaussianSplats({
  buffers,
}: {
  buffers: GaussianBuffers;
}) {
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [material, setMaterial] = React.useState<THREE.ShaderMaterial>();
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

    // Update component tate.
    setGeometry(geometry);
    setMaterial(new GaussianSplatMaterial());
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
      if (material !== undefined) material.dispose();
      sortWorker.postMessage({ close: true });
    };
  }, [buffers]);

  // Synchronize view projection matrix with sort worker.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const prevViewProj = React.useRef<THREE.Matrix4>();
  useFrame((state) => {
    const mesh = meshRef.current;
    const sortWorker = splatSortWorkerRef.current;
    if (mesh === null || sortWorker === null) return;

    // Update uniforms.
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
