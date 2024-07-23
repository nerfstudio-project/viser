import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import { GaussianSplatsContext } from "./SplatContext";
import { ViewerContext } from "../App";

const GaussianSplatMaterial = /* @__PURE__ */ shaderMaterial(
  {
    numGaussians: 0,
    focal: 100.0,
    viewport: [640, 480],
    near: 1.0,
    far: 100.0,
    depthTest: true,
    depthWrite: false,
    transparent: true,
    bufferTexture: null,
    groupTransformTexture: null,
    transitionInState: 0.0,
  },
  `precision highp usampler2D; // Most important: ints must be 32-bit.
  precision mediump float;

  // Index from the splat sorter.
  attribute uint sortedIndex;
  attribute uint groupIndex;

  // Buffers for splat data; each Gaussian gets 4 floats and 4 int32s. We just
  // copy quadjr for this.
  uniform usampler2D bufferTexture;
  uniform sampler2D groupTransformTexture;

  // Various other uniforms...
  uniform uint numGaussians;
  uniform vec2 focal;
  uniform vec2 viewport;
  uniform float near;
  uniform float far;

  // Fade in state between [0, 1].
  uniform float transitionInState;

  out vec4 vRgba;
  out vec2 vPosition;

	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}

  // Function to fetch and construct the i-th transform matrix using texelFetch
  mat4 getTransform(uint i) {
    // Calculate the base index for the i-th transform
    uint baseIndex = i * 3u;

    // Fetch the texels that represent the first 3 rows of the transform
    vec4 row0 = texelFetch(groupTransformTexture, ivec2(baseIndex + 0u, 0), 0);
    vec4 row1 = texelFetch(groupTransformTexture, ivec2(baseIndex + 1u, 0), 0);
    vec4 row2 = texelFetch(groupTransformTexture, ivec2(baseIndex + 2u, 0), 0);

    // Construct the mat4 with the fetched rows
    mat4 transform = mat4(
      row0,                 // First row
      row1,                 // Second row
      row2,                 // Third row
      vec4(0.0, 0.0, 0.0, 1.0)  // Fourth row is [0, 0, 0, 1] for SE(3)
    );
    return transpose(transform);
  }

  void main () {
    // Any early return will discard the fragment.
    gl_Position = vec4(0.0, 0.0, 2000.0, 1.0);

    // Get position + scale from float buffer.
    ivec2 texSize = textureSize(bufferTexture, 0);
    ivec2 texPos0 = ivec2((sortedIndex * 2u) % uint(texSize.x), (sortedIndex * 2u) / uint(texSize.x));
    uvec4 floatBufferData = texelFetch(bufferTexture, texPos0, 0);
    vec3 center = uintBitsToFloat(floatBufferData.xyz);

    mat4 objectTransform = getTransform(groupIndex);

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec4 c_cam = modelViewMatrix * objectTransform * vec4(center, 1);
    if (-c_cam.z < near || -c_cam.z > far)
      return;
    vec4 pos2d = projectionMatrix * c_cam;
    float clip = 1.1 * pos2d.w;
    if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip)
      return;

    float perGaussianShift = 1.0 - (float(numGaussians * 2u) - float(sortedIndex)) / float(numGaussians * 2u);
    float cov_scale = max(0.0, transitionInState - perGaussianShift) / (1.0 - perGaussianShift);

    // Get covariance terms from int buffer.
    ivec2 texPos1 = ivec2((sortedIndex * 2u + 1u) % uint(texSize.x), (sortedIndex * 2u + 1u) / uint(texSize.x));
    uvec4 intBufferData = texelFetch(bufferTexture, texPos1, 0);
    uint rgbaUint32 = intBufferData.w;
    vec2 chol01 = unpackHalf2x16(intBufferData.x);
    vec2 chol23 = unpackHalf2x16(intBufferData.y);
    vec2 chol45 = unpackHalf2x16(intBufferData.z);

    // Do the actual splatting.
    mat3 chol = mat3(
        chol01.x, chol01.y, chol23.x,
        0.,       chol23.y, chol45.x,
        0.,       0.,       chol45.y
    );
    mat3 cov3d = chol * transpose(chol) * cov_scale;
    mat3 J = mat3(
        // Matrices are column-major.
        focal.x / c_cam.z, 0., 0.0,
        0., focal.y / c_cam.z, 0.0,
        -(focal.x * c_cam.x) / (c_cam.z * c_cam.z), -(focal.y * c_cam.y) / (c_cam.z * c_cam.z), 0.
    );
    mat3 A = J * mat3(modelViewMatrix) * mat3(objectTransform);
    mat3 cov_proj = A * cov3d * transpose(A);
    float diag1 = cov_proj[0][0] + 0.3;
    float offDiag = cov_proj[0][1];
    float diag2 = cov_proj[1][1] + 0.3;

    // Eigendecomposition.
    float mid = 0.5 * (diag1 + diag2);
    float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;
    if (lambda2 < 0.0)
      return;
    vec2 diagonalVector = normalize(vec2(offDiag, lambda1 - diag1));
    vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vRgba = vec4(
      float(rgbaUint32 & uint(0xFF)) / 255.0,
      float((rgbaUint32 >> uint(8)) & uint(0xFF)) / 255.0,
      float((rgbaUint32 >> uint(16)) & uint(0xFF)) / 255.0,
      float(rgbaUint32 >> uint(24)) / 255.0
    );

    // Throw the Gaussian off the screen if it's too close, too far, or too small.
    float weightedDeterminant = vRgba.a * (diag1 * diag2 - offDiag * offDiag);
    if (weightedDeterminant < 0.1)
      return;
    if (weightedDeterminant < 1.0 && hash3D(center) < weightedDeterminant)  // This is not principled. It just makes things faster.
      return;
    vPosition = position.xy;

    gl_Position = vec4(
        vec2(pos2d) / pos2d.w
            + position.x * v1 / viewport * 2.0
            + position.y * v2 / viewport * 2.0, pos2d.z / pos2d.w, 1.);
  }
`,
  `precision mediump float;

  uniform vec2 viewport;
  uniform vec2 focal;

  in vec4 vRgba;
  in vec2 vPosition;

  void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vRgba.a;
    if (B < 0.01) discard;  // alphaTest.
    gl_FragColor = vec4(vRgba.rgb, B);
  }`,
);

export default function GlobalGaussianSplats() {
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [material, setMaterial] = React.useState<THREE.ShaderMaterial>();
  const splatSortWorkerRef = React.useRef<Worker | null>(null);
  const maxTextureSize = useThree((state) => state.gl).capabilities
    .maxTextureSize;
  const initializedTextures = React.useRef<boolean>(false);

  const viewer = React.useContext(ViewerContext)!;
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const allBuffers = splatContext((state) => state.bufferFromName);

  const [
    numGaussians,
    gaussianBuffer,
    numGroups,
    groupIndices,
    groupTransformBuffer,
  ] = React.useMemo(() => {
    // Create geometry. Each Gaussian will be rendered as a quad.
    let totalBufferLength = 0;
    for (const buffer of Object.values(allBuffers)) {
      totalBufferLength += buffer.length;
    }
    const numGaussians = totalBufferLength / 8;
    const gaussianBuffer = new Uint32Array(totalBufferLength);
    const groupIndices = new Uint32Array(numGaussians);

    let offset = 0;
    for (const [groupIndex, groupBuffer] of Object.values(
      allBuffers,
    ).entries()) {
      groupIndices.fill(
        groupIndex,
        offset / 8,
        (offset + groupBuffer.length) / 8,
      );
      gaussianBuffer.set(groupBuffer, offset);
      offset += groupBuffer.length;
    }

    const numGroups = Object.keys(allBuffers).length;
    const groupTransformBuffer = new Float32Array(numGroups * 12);
    return [
      numGaussians,
      gaussianBuffer,
      numGroups,
      groupIndices,
      groupTransformBuffer,
    ];
  }, [allBuffers]);

  const groupTransformTextureRef = React.useRef<THREE.DataTexture | null>(null);

  // We'll use the vanilla three.js API, which for our use case is more
  // flexible than the declarative version (particularly for operations like
  // dynamic updates to buffers and shader uniforms).
  React.useEffect(() => {
    if (numGaussians == 0) return;

    const geometry = new THREE.InstancedBufferGeometry();
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

    // Which group is each Gaussian in?
    const groupIndexAttribute = new THREE.InstancedBufferAttribute(
      groupIndices.slice(), // Copies the array.
      1,
    );
    groupIndexAttribute.setUsage(THREE.StaticDrawUsage);
    geometry.setAttribute("groupIndex", groupIndexAttribute);

    // Create texture buffers.
    const textureWidth = Math.min(numGaussians * 2, maxTextureSize);
    const textureHeight = Math.ceil((numGaussians * 2) / textureWidth);

    const bufferPadded = new Uint32Array(textureWidth * textureHeight * 4);
    bufferPadded.set(gaussianBuffer);
    const bufferTexture = new THREE.DataTexture(
      bufferPadded,
      textureWidth,
      textureHeight,
      THREE.RGBAIntegerFormat,
      THREE.UnsignedIntType,
    );
    bufferTexture.internalFormat = "RGBA32UI";

    const groupTransformTexture = new THREE.DataTexture(
      groupTransformBuffer,
      (numGroups * 12) / 4,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    );
    groupTransformTextureRef.current = groupTransformTexture;
    groupTransformTexture.internalFormat = "RGBA32F";

    const material = new GaussianSplatMaterial({
      // @ts-ignore
      bufferTexture: bufferTexture,
      groupTransformTexture: groupTransformTexture,
      numGaussians: 0,
      transitionInState: 0.0,
    });

    // Update component state.
    setGeometry(geometry);
    setMaterial(material);

    // Create sorting worker.
    const sortWorker = new SplatSortWorker();
    sortWorker.onmessage = (e) => {
      // Update the group + order index attributes.
      const sortedIndices = e.data.sortedIndices as Uint32Array;
      for (const [index, sortedIndex] of sortedIndices.entries()) {
        groupIndexAttribute.array[index] = groupIndices[sortedIndex];
      }
      sortedIndexAttribute.set(sortedIndices);

      groupIndexAttribute.needsUpdate = true;
      sortedIndexAttribute.needsUpdate = true;

      // Done sorting!
      isSortingRef.current = false;

      // Render Gaussians last.
      meshRef.current!.renderOrder = 1000.0;

      // Trigger initial render.
      if (!initializedTextures.current) {
        material.uniforms.numGaussians.value = numGaussians;
        bufferTexture.needsUpdate = true;
        initializedTextures.current = true;
      }
    };
    sortWorker.postMessage({
      setBuffer: gaussianBuffer,
      setGroupIndices: groupIndices,
    });
    splatSortWorkerRef.current = sortWorker;

    // We should always re-send view projection when buffers are replaced.
    prevT_camera_world.identity();

    return () => {
      bufferTexture.dispose();
      geometry.dispose();
      if (material !== undefined) material.dispose();
      sortWorker.postMessage({ close: true });
    };
  }, [numGaussians, gaussianBuffer, groupIndices]);

  // Synchronize view projection matrix with sort worker. We pre-allocate some
  // matrices to make life easier for the garbage collector.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const isSortingRef = React.useRef(false);
  const [prevT_camera_world] = React.useState(new THREE.Matrix4());

  useFrame((state, delta) => {
    const mesh = meshRef.current;
    const sortWorker = splatSortWorkerRef.current;
    if (mesh === null || sortWorker === null) return;

    // Update camera parameter uniforms.
    const dpr = state.viewport.dpr;
    const fovY =
      ((state.camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * state.viewport.aspect);
    const fy = (dpr * state.size.height) / (2 * Math.tan(fovY / 2));
    const fx = (dpr * state.size.width) / (2 * Math.tan(fovX / 2));

    if (material === undefined) return;

    const uniforms = material.uniforms;
    uniforms.transitionInState.value = Math.min(
      uniforms.transitionInState.value + delta * 2.0,
      1.0,
    );
    uniforms.focal.value = [fx, fy];
    uniforms.near.value = state.camera.near;
    uniforms.far.value = state.camera.far;
    uniforms.viewport.value = [state.size.width * dpr, state.size.height * dpr];

    // Compute view projection matrix.
    // T_camera_world = T_cam_world * T_world_obj.
    const T_camera_world = state.camera.matrixWorldInverse;
    // T_camera_world.copy(state.camera.matrixWorldInverse).multiply(
    //   mesh.matrixWorld,
    // );

    if (groupTransformTextureRef.current !== null) {
      for (const [groupIndex, name] of Object.keys(allBuffers).entries()) {
        const node = viewer.nodeRefFromName.current[name];
        if (node === undefined) continue;
        const rowMajorElements = node.matrixWorld
          .transpose()
          .elements.slice(0, 12);
        groupTransformBuffer.set(rowMajorElements, groupIndex * 12);
      }
      groupTransformTextureRef.current.needsUpdate = true;
      sortWorker.postMessage({ setT_world_objs: groupTransformBuffer });
    }

    // If changed, use projection matrix to sort Gaussians.
    if (
      !isSortingRef.current &&
      (prevT_camera_world === undefined ||
        !T_camera_world.equals(prevT_camera_world))
    ) {
      sortWorker.postMessage({ setT_camera_world: T_camera_world.elements });
      isSortingRef.current = true;
      prevT_camera_world.copy(T_camera_world);
    }
  });

  return <mesh ref={meshRef} geometry={geometry} material={material} />;
}
