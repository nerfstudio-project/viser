/** Gaussian splatting implementation for viser.
 *
 * This borrows heavily from existing open-source implementations. Particularly
 * useful references:
 * - https://github.com/quadjr/aframe-gaussian-splatting
 * - https://github.com/antimatter15/splat
 * - https://github.com/pmndrs/drei
 * - https://github.com/vincent-lecrubier-skydio/react-three-fiber-gaussian-splat
 *
 * Usage should look like:
 *
 * <Canvas>
 *   <SplatRenderContext>
 *     <SplatObject buffer={buffer} />
 *   </SplatRenderContext>
 * </Canvas>
 *
 * Where `buffer` contains serialized Gaussian attributes. SplatObjects are
 * globally sorted by a worker (with some help from WebAssembly + SIMD
 * intrinsics), and then rendered as a single threejs mesh. Unlike other R3F
 * implementations that we're aware of, this enables correct compositing
 * between multiple splat objects.
 */

import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import { SorterWorkerIncoming } from "./SplatSortWorker";
import { create } from "zustand";
import { Object3D } from "three";

/**Global splat state.*/
interface SplatState {
  groupBufferFromId: { [id: string]: Uint32Array };
  nodeRefFromId: React.MutableRefObject<{
    [name: string]: undefined | Object3D;
  }>;
  setBuffer: (id: string, buffer: Uint32Array) => void;
  removeBuffer: (id: string) => void;
}

/**Hook for creating global splat state.*/
function useGaussianSplatStore() {
  const nodeRefFromId = React.useRef({});
  return React.useState(() =>
    create<SplatState>((set) => ({
      groupBufferFromId: {},
      nodeRefFromId: nodeRefFromId,
      setBuffer: (id, buffer) => {
        return set((state) => ({
          groupBufferFromId: { ...state.groupBufferFromId, [id]: buffer },
        }));
      },
      removeBuffer: (id) => {
        return set((state) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { [id]: _, ...buffers } = state.groupBufferFromId;
          return { groupBufferFromId: buffers };
        });
      },
    })),
  )[0];
}
const GaussianSplatsContext = React.createContext<ReturnType<
  typeof useGaussianSplatStore
> | null>(null);

/**Provider for creating splat rendering context.*/
export function SplatRenderContext({
  children,
}: {
  children: React.ReactNode;
}) {
  const store = useGaussianSplatStore();
  return (
    <GaussianSplatsContext.Provider value={store}>
      <SplatRenderer />
      {children}
    </GaussianSplatsContext.Provider>
  );
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
    textureBuffer: null,
    textureT_camera_groups: null,
    transitionInState: 0.0,
  },
  `precision highp usampler2D; // Most important: ints must be 32-bit.
  precision mediump float;

  // Index from the splat sorter.
  attribute uint sortedIndex;

  // Buffers for splat data; each Gaussian gets 4 floats and 4 int32s. We just
  // copy quadjr for this.
  uniform usampler2D textureBuffer;

  // We could also use a uniform to store transforms, but this would be more
  // limiting in terms of the # of groups we can have.
  uniform sampler2D textureT_camera_groups;

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

  // Function to fetch and construct the i-th transform matrix using texelFetch
  mat4 getGroupTransform(uint i) {
    // Calculate the base index for the i-th transform.
    uint baseIndex = i * 3u;

    // Fetch the texels that represent the first 3 rows of the transform. We
    // choose to use row-major here, since it lets us exclude the fourth row of
    // the matrix.
    vec4 row0 = texelFetch(textureT_camera_groups, ivec2(baseIndex + 0u, 0), 0);
    vec4 row1 = texelFetch(textureT_camera_groups, ivec2(baseIndex + 1u, 0), 0);
    vec4 row2 = texelFetch(textureT_camera_groups, ivec2(baseIndex + 2u, 0), 0);

    // Construct the mat4 with the fetched rows.
    mat4 transform = mat4(row0, row1, row2, vec4(0.0, 0.0, 0.0, 1.0));
    return transpose(transform);
  }

  void main () {
    // Get position + scale from float buffer.
    ivec2 texSize = textureSize(textureBuffer, 0);
    uint texStart = sortedIndex << 1u;
    ivec2 texPos0 = ivec2(texStart % uint(texSize.x), texStart / uint(texSize.x));


    // Fetch from textures.
    uvec4 floatBufferData = texelFetch(textureBuffer, texPos0, 0);
    mat4 T_camera_group = getGroupTransform(floatBufferData.w);

    // Any early return will discard the fragment.
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec3 center = uintBitsToFloat(floatBufferData.xyz);
    vec4 c_cam = T_camera_group * vec4(center, 1);
    if (-c_cam.z < near || -c_cam.z > far)
      return;
    vec4 pos2d = projectionMatrix * c_cam;
    float clip = 1.1 * pos2d.w;
    if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip)
      return;

    // Read covariance terms.
    ivec2 texPos1 = ivec2((texStart + 1u) % uint(texSize.x), (texStart + 1u) / uint(texSize.x));
    uvec4 intBufferData = texelFetch(textureBuffer, texPos1, 0);

    // Get covariance terms from int buffer.
    uint rgbaUint32 = intBufferData.w;
    vec2 chol01 = unpackHalf2x16(intBufferData.x);
    vec2 chol23 = unpackHalf2x16(intBufferData.y);
    vec2 chol45 = unpackHalf2x16(intBufferData.z);

    // Transition in.
    float startTime = 0.8 * float(sortedIndex) / float(numGaussians);
    float cov_scale = smoothstep(startTime, startTime + 0.2, transitionInState);

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
    mat3 A = J * mat3(T_camera_group);
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
    if (weightedDeterminant < 0.5)
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

export const SplatObject = React.forwardRef<
  THREE.Group,
  {
    buffer: Uint32Array;
  }
>(function SplatObject({ buffer }, ref) {
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const setBuffer = splatContext((state) => state.setBuffer);
  const removeBuffer = splatContext((state) => state.removeBuffer);
  const nodeRefFromId = splatContext((state) => state.nodeRefFromId);
  const name = React.useMemo(() => crypto.randomUUID(), [buffer]);

  const [obj, setRef] = React.useState<THREE.Group | null>(null);

  React.useEffect(() => {
    if (obj === null) return;
    setBuffer(name, buffer);
    if (ref !== null) {
      if ("current" in ref) {
        ref.current = obj;
      } else {
        ref(obj);
      }
    }
    nodeRefFromId.current[name] = obj;
    return () => {
      removeBuffer(name);
      delete nodeRefFromId.current[name];
    };
  }, [obj]);

  return <group ref={setRef}></group>;
});

/** External interface. Component should be added to the root of canvas.  */
function SplatRenderer() {
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const groupBufferFromId = splatContext((state) => state.groupBufferFromId);
  const nodeRefFromId = splatContext((state) => state.nodeRefFromId);

  // Consolidate Gaussian groups into a single buffer.
  const merged = mergeGaussianGroups(groupBufferFromId);
  const meshProps = useGaussianMeshProps(
    merged.gaussianBuffer,
    merged.numGroups,
  );

  // Create sorting worker.
  const sortWorker = new SplatSortWorker();
  let initializedBufferTexture = false;
  sortWorker.onmessage = (e) => {
    // Update rendering order.
    const sortedIndices = e.data.sortedIndices as Uint32Array;
    meshProps.sortedIndexAttribute.set(sortedIndices);
    meshProps.sortedIndexAttribute.needsUpdate = true;

    // Trigger initial render.
    if (!initializedBufferTexture) {
      meshProps.material.uniforms.numGaussians.value = merged.numGaussians;
      meshProps.textureBuffer.needsUpdate = true;
      initializedBufferTexture = true;
    }
  };
  function postToWorker(message: SorterWorkerIncoming) {
    sortWorker.postMessage(message);
  }

  postToWorker({
    setBuffer: merged.gaussianBuffer,
    setGroupIndices: merged.groupIndices,
  });

  // Cleanup.
  React.useEffect(() => {
    return () => {
      meshProps.textureBuffer.dispose();
      meshProps.geometry.dispose();
      meshProps.material.dispose();
      postToWorker({ close: true });
    };
  });

  // Per-frame updates. This is in charge of synchronizing transforms and
  // triggering sorting.
  //
  // We pre-allocate matrices to make life easier for the garbage collector.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const tmpT_camera_group = new THREE.Matrix4();
  const Tz_camera_groups = new Float32Array(merged.numGroups * 4);
  const prevRowMajorT_camera_groups = meshProps.rowMajorT_camera_groups
    .slice()
    .fill(0);
  const prevVisibles: boolean[] = [];
  useFrame((state, delta) => {
    const mesh = meshRef.current;
    if (mesh === null || sortWorker === null) return;

    // Update camera parameter uniforms.
    const dpr = state.viewport.dpr;
    const fovY =
      ((state.camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * state.viewport.aspect);
    const fy = (dpr * state.size.height) / (2 * Math.tan(fovY / 2));
    const fx = (dpr * state.size.width) / (2 * Math.tan(fovX / 2));

    if (meshProps.material === undefined) return;

    const uniforms = meshProps.material.uniforms;
    uniforms.transitionInState.value = Math.min(
      uniforms.transitionInState.value + delta * 2.0,
      1.0,
    );
    uniforms.focal.value = [fx, fy];
    uniforms.near.value = state.camera.near;
    uniforms.far.value = state.camera.far;
    uniforms.viewport.value = [state.size.width * dpr, state.size.height * dpr];

    // Update group transforms.
    const T_camera_world = state.camera.matrixWorldInverse;
    const groupVisibles: boolean[] = [];
    let visibilitiesChanged = false;
    for (const [groupIndex, name] of Object.keys(groupBufferFromId).entries()) {
      const node = nodeRefFromId.current[name];
      if (node === undefined) continue;
      tmpT_camera_group.copy(T_camera_world).multiply(node.matrixWorld);
      const colMajorElements = tmpT_camera_group.elements;
      Tz_camera_groups.set(
        [
          colMajorElements[2],
          colMajorElements[6],
          colMajorElements[10],
          colMajorElements[14],
        ],
        groupIndex * 4,
      );
      const rowMajorElements = tmpT_camera_group.transpose().elements;
      meshProps.rowMajorT_camera_groups.set(
        rowMajorElements.slice(0, 12),
        groupIndex * 12,
      );

      // Determine visibility. If the parent has unmountWhenInvisible=true, the
      // first frame after showing a hidden parent can have visible=true with
      // an incorrect matrixWorld transform. There might be a better fix, but
      // `prevVisible` is an easy workaround for this.
      let visibleNow = node.visible && node.parent !== null;
      if (visibleNow) {
        node.traverseAncestors((ancestor) => {
          visibleNow = visibleNow && ancestor.visible;
        });
      }
      groupVisibles.push(visibleNow && prevVisibles[groupIndex] === true);
      if (prevVisibles[groupIndex] !== visibleNow) {
        prevVisibles[groupIndex] = visibleNow;
        visibilitiesChanged = true;
      }
    }

    const groupsMovedWrtCam = !meshProps.rowMajorT_camera_groups.every(
      (v, i) => v === prevRowMajorT_camera_groups[i],
    );

    if (groupsMovedWrtCam) {
      // Gaussians need to be re-sorted.
      postToWorker({
        setTz_camera_groups: Tz_camera_groups,
      });
    }
    if (groupsMovedWrtCam || visibilitiesChanged) {
      // If a group is not visible, we'll throw it off the screen with some Big
      // Numbers. It's important that this only impacts the coordinates used
      // for the shader and not for the sorter; that way when we "show" a group
      // of Gaussians the correct rendering order is immediately available.
      for (const [i, visible] of groupVisibles.entries()) {
        if (!visible) {
          meshProps.rowMajorT_camera_groups[i * 12 + 3] = 1e10;
          meshProps.rowMajorT_camera_groups[i * 12 + 7] = 1e10;
          meshProps.rowMajorT_camera_groups[i * 12 + 11] = 1e10;
        }
      }
      prevRowMajorT_camera_groups.set(meshProps.rowMajorT_camera_groups);
      meshProps.textureT_camera_groups.needsUpdate = true;
    }
  }, -100 /* This should be called early to reduce group transform artifacts. */);

  return (
    <mesh
      ref={meshRef}
      geometry={meshProps.geometry}
      material={meshProps.material}
      renderOrder={10000.0 /*Generally, we want to render last.*/}
    />
  );
}

/**Consolidate groups of Gaussians into a single buffer, to make it possible
 * for them to be sorted globally.*/
function mergeGaussianGroups(groupBufferFromName: {
  [name: string]: Uint32Array;
}) {
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

    // Each Gaussian is allocated
    // - 12 bytes for center x, y, z (float32)
    // - 4 bytes for group index (uint32); we're filling this in now
    //
    // - 12 bytes for covariance (6 terms, float16)
    // - 4 bytes for RGBA (uint8)
    for (let i = 0; i < groupBuffer.length; i += 8) {
      gaussianBuffer[offset + i + 3] = groupIndex;
    }
    offset += groupBuffer.length;
  }

  const numGroups = Object.keys(groupBufferFromName).length;
  return { numGaussians, gaussianBuffer, numGroups, groupIndices };
}

/**Hook to generate properties for rendering Gaussians via a three.js mesh.*/
function useGaussianMeshProps(gaussianBuffer: Uint32Array, numGroups: number) {
  const numGaussians = gaussianBuffer.length / 8;
  const maxTextureSize = useThree((state) => state.gl).capabilities
    .maxTextureSize;

  // Create instanced geometry.
  const geometry = new THREE.InstancedBufferGeometry();
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

  // Rendering order for Gaussians.
  const sortedIndexAttribute = new THREE.InstancedBufferAttribute(
    new Uint32Array(numGaussians),
    1,
  );
  sortedIndexAttribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("sortedIndex", sortedIndexAttribute);

  // Create texture buffers.
  const textureWidth = Math.min(numGaussians * 2, maxTextureSize);
  const textureHeight = Math.ceil((numGaussians * 2) / textureWidth);
  const bufferPadded = new Uint32Array(textureWidth * textureHeight * 4);
  bufferPadded.set(gaussianBuffer);
  const textureBuffer = new THREE.DataTexture(
    bufferPadded,
    textureWidth,
    textureHeight,
    THREE.RGBAIntegerFormat,
    THREE.UnsignedIntType,
  );
  textureBuffer.internalFormat = "RGBA32UI";
  textureBuffer.needsUpdate = true;

  const rowMajorT_camera_groups = new Float32Array(numGroups * 12);
  const textureT_camera_groups = new THREE.DataTexture(
    rowMajorT_camera_groups,
    (numGroups * 12) / 4,
    1,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  textureT_camera_groups.internalFormat = "RGBA32F";
  textureT_camera_groups.needsUpdate = true;

  const material = new GaussianSplatMaterial({
    // @ts-ignore
    textureBuffer: textureBuffer,
    textureT_camera_groups: textureT_camera_groups,
    numGaussians: 0,
    transitionInState: 0.0,
  });
  console.log(numGaussians) // 447703
  console.log(textureWidth, textureWidth) // 16384 16384
  console.log(gaussianBuffer) // long list of ints
  return {
    geometry,
    material,
    textureBuffer,
    sortedIndexAttribute,
    textureT_camera_groups,
    rowMajorT_camera_groups,
  };
}
