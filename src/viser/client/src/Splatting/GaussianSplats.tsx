import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import { GaussianSplatsContext } from "./SplatContext";
import { ViewerContext } from "../App";
import { SorterWorkerIncoming } from "./SplatSortWorker";

function postToWorker(worker: Worker, message: SorterWorkerIncoming) {
  worker.postMessage(message);
}

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

  // Which group transform should be applied to each Gaussian.
  attribute uint groupIndex;

  // Buffers for splat data; each Gaussian gets 4 floats and 4 int32s. We just
  // copy quadjr for this.
  uniform usampler2D bufferTexture;

  // Buffer containing SE(3) transforms, which will be independently applied to
  // Gaussians. This enables multiple fast rigid "objects".
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
  mat4 getGroupTransform(uint i) {
    // Calculate the base index for the i-th transform.
    uint baseIndex = i * 3u;

    // Fetch the texels that represent the first 3 rows of the transform. We
    // choose to use row-major here, since it lets us exclude the fourth row of
    // the matrix.
    vec4 row0 = texelFetch(groupTransformTexture, ivec2(baseIndex + 0u, 0), 0);
    vec4 row1 = texelFetch(groupTransformTexture, ivec2(baseIndex + 1u, 0), 0);
    vec4 row2 = texelFetch(groupTransformTexture, ivec2(baseIndex + 2u, 0), 0);

    // Construct the mat4 with the fetched rows.
    mat4 transform = mat4(row0, row1, row2, vec4(0.0, 0.0, 0.0, 1.0));
    return transpose(transform);
  }

  void main () {
    // Get position + scale from float buffer.
    ivec2 texSize = textureSize(bufferTexture, 0);
    ivec2 texPos0 = ivec2((sortedIndex * 2u) % uint(texSize.x), (sortedIndex * 2u) / uint(texSize.x));

    // Fetch from textures.
    uvec4 floatBufferData = texelFetch(bufferTexture, texPos0, 0);
    mat4 groupTransform = getGroupTransform(groupIndex);

    // Any early return will discard the fragment.
    gl_Position = vec4(0.0, 0.0, 2000.0, 1.0);

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec3 center = uintBitsToFloat(floatBufferData.xyz);
    mat4 T_world_group = modelViewMatrix * groupTransform;
    vec4 c_cam = T_world_group * vec4(center, 1);
    if (-c_cam.z < near || -c_cam.z > far)
      return;
    vec4 pos2d = projectionMatrix * c_cam;
    float clip = 1.1 * pos2d.w;
    if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip)
      return;

    // Read covariance terms.
    ivec2 texPos1 = ivec2((sortedIndex * 2u + 1u) % uint(texSize.x), (sortedIndex * 2u + 1u) / uint(texSize.x));
    uvec4 intBufferData = texelFetch(bufferTexture, texPos1, 0);

    // Get covariance terms from int buffer.
    uint rgbaUint32 = intBufferData.w;
    vec2 chol01 = unpackHalf2x16(intBufferData.x);
    vec2 chol23 = unpackHalf2x16(intBufferData.y);
    vec2 chol45 = unpackHalf2x16(intBufferData.z);

    // Transition in.
    float perGaussianShift = 1.0 - (float(numGaussians * 2u) - float(sortedIndex)) / float(numGaussians * 2u);
    float cov_scale = max(0.0, transitionInState - perGaussianShift) / (1.0 - perGaussianShift);

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
    mat3 A = J * mat3(T_world_group);
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
  const initializedBufferTexture = React.useRef<boolean>(false);
  const groupTransformTextureRef = React.useRef<THREE.DataTexture | null>(null);

  const viewer = React.useContext(ViewerContext)!;
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const groupBufferFromName = splatContext(
    (state) => state.groupBufferFromName,
  );

  // Pre-compute some buffers we need for rendering.
  const [
    numGaussians,
    gaussianBuffer,
    numGroups,
    groupIndices,
    groupTransformBuffer,
  ] = React.useMemo(() => {
    // Create geometry. Each Gaussian will be rendered as a quad.
    let totalBufferLength = 0;
    for (const buffer of Object.values(groupBufferFromName)) {
      totalBufferLength += buffer.length;
    }
    const numGaussians = totalBufferLength / 8;
    const gaussianBuffer = new Uint32Array(totalBufferLength);
    const groupIndices = new Uint32Array(numGaussians);

    let offset = 0;
    for (const [groupIndex, groupBuffer] of Object.values(
      groupBufferFromName,
    ).entries()) {
      groupIndices.fill(
        groupIndex,
        offset / 8,
        (offset + groupBuffer.length) / 8,
      );
      gaussianBuffer.set(groupBuffer, offset);
      offset += groupBuffer.length;
    }

    const numGroups = Object.keys(groupBufferFromName).length;
    const groupTransformBuffer = new Float32Array(numGroups * 12);
    return [
      numGaussians,
      gaussianBuffer,
      numGroups,
      groupIndices,
      groupTransformBuffer,
    ];
  }, [groupBufferFromName]);

  // Primary setup.
  //
  // We'll use the vanilla three.js API, which for our use case is more
  // flexible than the declarative version (particularly for operations like
  // dynamic updates to buffers and shader uniforms).
  React.useEffect(() => {
    if (numGaussians == 0) {
      return;
    }

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
    bufferTexture.needsUpdate = true;

    const groupTransformTexture = new THREE.DataTexture(
      groupTransformBuffer,
      (numGroups * 12) / 4,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    );
    groupTransformTexture.internalFormat = "RGBA32F";
    groupTransformTextureRef.current = groupTransformTexture;
    groupTransformTexture.needsUpdate = true;

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
      // Update rendering order.
      const sortedIndices = e.data.sortedIndices as Uint32Array;
      sortedIndexAttribute.set(sortedIndices);
      sortedIndexAttribute.needsUpdate = true;

      // Update group indices if needed.
      if (numGroups >= 2) {
        const sortedGroupIndices = e.data.sortedGroupIndices as Uint32Array;
        groupIndexAttribute.set(sortedGroupIndices);
        groupIndexAttribute.needsUpdate = true;
      }

      // Trigger initial render.
      if (!initializedBufferTexture.current) {
        material.uniforms.numGaussians.value = numGaussians;
        bufferTexture.needsUpdate = true;
        initializedBufferTexture.current = true;
      }
    };
    postToWorker(sortWorker, {
      setBuffer: gaussianBuffer,
      setGroupIndices: groupIndices,
    });
    splatSortWorkerRef.current = sortWorker;

    // We should always re-send view projection when buffers are replaced.
    prevT_camera_world.identity();

    return () => {
      bufferTexture.dispose();
      groupTransformTexture.dispose();
      groupTransformTextureRef.current = null;
      geometry.dispose();
      if (material !== undefined) material.dispose();
      postToWorker(sortWorker, { close: true });
      splatSortWorkerRef.current = null;
    };
  }, [numGaussians, gaussianBuffer, groupIndices]);

  // Per-frame updates. This is in charge of synchronizing transforms and
  // triggering sorting.
  //
  // Synchronize view projection matrix with sort worker. We pre-allocate some
  // matrices to make life easier for the garbage collector.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const [prevT_camera_world] = React.useState(new THREE.Matrix4());
  const tmpGroupTransformBuffer = React.useMemo(
    () => new Float32Array(numGroups * 12),
    [numGroups],
  );
  const staleSort = React.useRef(false);
  useFrame((state, delta) => {
    const mesh = meshRef.current;
    const sortWorker = splatSortWorkerRef.current;
    if (
      mesh === null ||
      sortWorker === null ||
      groupTransformTextureRef.current == null
    )
      return;

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

    // Update group transforms.
    if (groupTransformTextureRef.current !== null) {
      for (const [groupIndex, name] of Object.keys(
        groupBufferFromName,
      ).entries()) {
        const node = viewer.nodeRefFromName.current[name];
        if (node === undefined) continue;
        const rowMajorElements = node.matrixWorld
          .transpose()
          .elements.slice(0, 12);
        tmpGroupTransformBuffer.set(rowMajorElements, groupIndex * 12);

        // If a group is not visible, we'll throw it off the screen with some
        // Big Numbers.
        let visible = node.visible && node.parent !== null;
        if (visible) {
          node.traverseAncestors((ancestor) => {
            visible = visible && ancestor.visible;
          });
        }
        if (!visible) {
          tmpGroupTransformBuffer[groupIndex * 12 + 3] = 1e10;
          tmpGroupTransformBuffer[groupIndex * 12 + 7] = 1e10;
          tmpGroupTransformBuffer[groupIndex * 12 + 11] = 1e10;
        }
      }
      if (
        !groupTransformBuffer.every((v, i) => v === tmpGroupTransformBuffer[i])
      ) {
        staleSort.current = true;
        groupTransformBuffer.set(tmpGroupTransformBuffer);
        groupTransformTextureRef.current.needsUpdate = true;
        postToWorker(sortWorker, {
          // Big values will break the counting sort precision. We just
          // zero them out for now.
          setT_world_groups: groupTransformBuffer.map((x) =>
            x >= 1e10 ? 0.0 : x,
          ),
        });
      }
    }

    // Update camera transform.
    const T_camera_world = state.camera.matrixWorldInverse;

    if (!T_camera_world.equals(prevT_camera_world)) {
      staleSort.current = true;
      postToWorker(sortWorker, {
        setTz_camera_world: [
          T_camera_world.elements[2],
          T_camera_world.elements[6],
          T_camera_world.elements[10],
          T_camera_world.elements[14],
        ],
      });
      prevT_camera_world.copy(T_camera_world);
    }

    // Sort Gaussians.
    if (staleSort.current) {
      staleSort.current = false;
      postToWorker(sortWorker, {
        triggerSort: true,
      });
    }
  });

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      material={material}
      renderOrder={10000.0 /*Generally, we want to render last.*/}
    />
  );
}
