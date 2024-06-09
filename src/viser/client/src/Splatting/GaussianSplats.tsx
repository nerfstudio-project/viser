import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";

export type GaussianBuffers = {
  // See: https://github.com/quadjr/aframe-gaussian-splatting
  //
  // - x as f32
  // - y as f32
  // - z as f32
  // - cov scale as f32
  floatBuffer: Float32Array;
  // cov1 (int16), cov2 (int16) packed in int32
  // cov3 (int16), cov4 (int16) packed in int32
  // cov5 (int16), cov6 (int16) packed in int32
  // rgba packed in int32
  intBuffer: Int32Array;
};

const GaussianSplatMaterial = /* @__PURE__ */ shaderMaterial(
  {
    alphaTest: 0.05,
    alphaHash: false,
    numGaussians: 0,
    focal: 100.0,
    viewport: [640, 480],
    depthTest: true,
    depthWrite: true,
    transparent: true,
    floatBufferTexture: null,
    intBufferTexture: null,
  },
  `precision highp usampler2D; // Most important: ints must be 32-bit.
  precision mediump sampler2D;

  attribute uint sortedIndex;

  uniform sampler2D floatBufferTexture;
  uniform usampler2D intBufferTexture;
  uniform uint numGaussians;
  uniform vec2 focal;
  uniform vec2 viewport;

  varying vec4 vRgba;
  varying float vOpacity;
  varying vec2 vPosition;

  vec2 unpackInt16(in uint value) {
    int v = int(value);
    int v0 = v >> 16;
    int v1 = (v & 0xFFFF);
    if((v & 0x8000) != 0)
      v1 |= 0xFFFF0000;
    return vec2(float(v1), float(v0));
  }

  void main () {
    // Read from textures.
    ivec2 texSize = textureSize(floatBufferTexture, 0);
    ivec2 texPos = ivec2(sortedIndex % uint(texSize.x), sortedIndex / uint(texSize.x));
    vec4 floatBufferData = texelFetch(floatBufferTexture, texPos, 0);
    vec3 center = floatBufferData.xyz;
    float cov_scale = floatBufferData.w;

    uvec4 intBufferData = texelFetch(intBufferTexture, texPos, 0);
    uint rgbaUint32 = intBufferData.w;
    vec2 cov01 = unpackInt16(intBufferData.x) / 32767. * cov_scale;
    vec2 cov23 = unpackInt16(intBufferData.y) / 32767. * cov_scale;
    vec2 cov45 = unpackInt16(intBufferData.z) / 32767. * cov_scale;

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec4 c_cam = modelViewMatrix * vec4(center, 1);
    vec4 pos2d = projectionMatrix * c_cam;

    // Splat covariance.
    mat3 cov3d = mat3(
        cov01.x, cov01.y, cov23.x,
        cov01.y, cov23.y, cov45.x,
        cov23.x, cov45.x, cov45.y
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

    // Eigendecomposition.
    float mid = 0.5 * (diag1 + diag2);
    float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    vec2 diagonalVector = normalize(vec2(offDiag, lambda1 - diag1));
    vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vRgba = vec4(
      float(rgbaUint32 & uint(0xFF)) / 255.0,
      float((rgbaUint32 >> uint(8)) & uint(0xFF)) / 255.0,
      float((rgbaUint32 >> uint(16)) & uint(0xFF)) / 255.0,
      float(rgbaUint32 >> uint(24)) / 255.0
    );
    vPosition = vec2(position.x, position.y);

    gl_Position = vec4(
        vec2(pos2d) / pos2d.w
            + position.x * v1 / viewport * 2.0
            + position.y * v2 / viewport * 2.0, pos2d.z / pos2d.w, 1.);
  }
`,
  `
  #include <alphatest_pars_fragment>
  #include <alphahash_pars_fragment>

  precision highp float;

  uniform vec2 viewport;
  uniform vec2 focal;

  varying vec4 vRgba;
  varying vec2 vPosition;


  void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vRgba.a;
    vec4 diffuseColor = vec4(vRgba.rgb, B);
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
  const splatSortWorkerRef = React.useRef<Worker | null>(null);

  const gl = useThree((state) => state.gl);
  const maxTextureSize = gl.capabilities.maxTextureSize;
  const originRef = React.useRef<THREE.Vector3>(new THREE.Vector3());

  // We'll use the vanilla three.js API, which for our use case is more
  // flexible than the declarative version (particularly for operations like
  // dynamic updates to buffers and shader uniforms).
  React.useEffect(() => {
    // Create geometry. Each Gaussian will be rendered as a quad.
    const geometry = new THREE.InstancedBufferGeometry();
    const numGaussians = buffers.floatBuffer.length / 4;
    geometry.instanceCount = numGaussians;

    // Quad geometry.
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

    // Rendering order for Gaussians.
    const sortedIndexAttribute = new THREE.InstancedBufferAttribute(
      new Uint32Array(numGaussians),
      1,
    );
    sortedIndexAttribute.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("sortedIndex", sortedIndexAttribute);

    // Compute center of the Gaussians. This is used for a render ordering
    // heuristic.
    for (let i = 0; i < numGaussians; i++) {
      originRef.current.x += buffers.floatBuffer[i * 3 + 0];
      originRef.current.y += buffers.floatBuffer[i * 3 + 1];
      originRef.current.z += buffers.floatBuffer[i * 3 + 2];
    }
    originRef.current.divideScalar(numGaussians);

    const textureWidth = Math.min(numGaussians, maxTextureSize);
    const textureHeight = Math.ceil(numGaussians / textureWidth);
    const floatBufferPadded = new Float32Array(
      textureWidth * textureHeight * 4,
    );
    floatBufferPadded.set(buffers.floatBuffer);
    const intBufferPadded = new Uint32Array(textureWidth * textureHeight * 4);
    intBufferPadded.set(buffers.intBuffer);
    const floatBufferTexture = new THREE.DataTexture(
      floatBufferPadded,
      textureWidth,
      textureHeight,
      THREE.RGBAFormat,
      THREE.FloatType,
    );

    const intBufferTexture = new THREE.DataTexture(
      intBufferPadded,
      textureWidth,
      textureHeight,
      THREE.RGBAIntegerFormat,
      THREE.UnsignedIntType,
    );
    intBufferTexture.internalFormat = "RGBA32UI";

    const material = new GaussianSplatMaterial({
      // @ts-ignore
      floatBufferTexture: floatBufferTexture,
      intBufferTexture: intBufferTexture,
      numGaussians: numGaussians,
    });

    // Update component state.
    setGeometry(geometry);
    setMaterial(material);

    // Create sorting worker.
    const sortWorker = new SplatSortWorker();
    sortWorker.onmessage = (e) => {
      sortedIndexAttribute.set(e.data.sortedIndices as Int32Array);
      sortedIndexAttribute.needsUpdate = true;
      if (!initializedTextures.current) {
        floatBufferTexture.needsUpdate = true;
        intBufferTexture.needsUpdate = true;
        initializedTextures.current = true;
      }
    };
    sortWorker.postMessage({
      setFloatBuffer: buffers.floatBuffer,
    });
    splatSortWorkerRef.current = sortWorker;

    // We should always re-send view projection when buffers are replaced.
    prevViewProj.identity();

    return () => {
      // intBufferTexture.dispose();
      floatBufferTexture.dispose();
      geometry.dispose();
      if (material !== undefined) material.dispose();
      sortWorker.postMessage({ close: true });
    };
  }, [buffers]);

  // Synchronize view projection matrix with sort worker.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const initializedTextures = React.useRef<boolean>(false);
  const prevViewProj = new THREE.Matrix4();
  const viewProj = new THREE.Matrix4();
  const T_camera_world = new THREE.Matrix4();

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
    T_camera_world.copy(state.camera.matrixWorldInverse).multiply(
      mesh.matrixWorld,
    );
    viewProj.copy(state.camera.projectionMatrix).multiply(T_camera_world);

    // Heuristic for rendering order; reduces artifacts when multiple splat
    // objects are present.
    meshRef.current!.renderOrder = originRef.current
      .clone()
      .applyMatrix4(T_camera_world).z;

    // If changed, use projection matrix to sort Gaussians.
    if (prevViewProj === undefined || !viewProj.equals(prevViewProj)) {
      sortWorker.postMessage({ setViewProj: viewProj.elements });
      prevViewProj.copy(viewProj);
    }
  });

  return <mesh ref={meshRef} geometry={geometry} material={material} />;
}
