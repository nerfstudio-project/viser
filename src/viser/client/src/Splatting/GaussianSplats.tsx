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

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import { SorterWorkerIncoming } from "./SplatSortWorker";
import { create } from "zustand";
import { Object3D } from "three";
import { v4 as uuidv4 } from "uuid";

/**Global splat state.*/
interface SplatState {
  groupBufferFromId: {
    [id: string]: [Uint32Array, Uint32Array, number];
  };
  nodeRefFromId: React.MutableRefObject<{
    [name: string]: undefined | Object3D;
  }>;
  setBuffers: (id: string, buffers: [Uint32Array, Uint32Array, number]) => void;
  removeBuffers: (id: string) => void;
}

/**Hook for creating global splat state.*/
function useGaussianSplatStore() {
  const nodeRefFromId = React.useRef({});
  return React.useState(() =>
    create<SplatState>((set) => ({
      groupBufferFromId: {},
      nodeRefFromId: nodeRefFromId,
      setBuffers: (id, buffer) => {
        return set((state) => ({
          groupBufferFromId: { ...state.groupBufferFromId, [id]: buffer },
        }));
      },
      removeBuffers: (id) => {
        return set((state) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { [id]: _, ...buffers } = state.groupBufferFromId;
          return { groupBufferFromId: buffers };
        });
      },
    })),
  )[0];
}

export const GaussianSplatsContext = React.createContext<{
  postToWorkerRef: React.MutableRefObject<
    (message: SorterWorkerIncoming) => void
  >;
  materialRef: React.MutableRefObject<THREE.ShaderMaterial | null>;
  shapeOfMotionBasesRef: React.MutableRefObject<Float32Array>;
  useGaussianSplatStore: ReturnType<typeof useGaussianSplatStore>;
  updateCamera: React.MutableRefObject<
    | null
    | ((
        camera: THREE.PerspectiveCamera,
        width: number,
        height: number,
        blockingSort: boolean,
      ) => void)
  >;
  meshPropsRef: React.MutableRefObject<ReturnType<
    typeof useGaussianMeshProps
  > | null>;
} | null>(null);

/**Provider for creating splat rendering context.*/
export function SplatRenderContext({
  children,
}: {
  children: React.ReactNode;
}) {
  const store = useGaussianSplatStore();
  return (
    <GaussianSplatsContext.Provider
      value={{
        postToWorkerRef: React.useRef((message) => {}),
        materialRef: React.useRef(null),
        shapeOfMotionBasesRef: React.useRef(new Float32Array(30 * 4)),
        useGaussianSplatStore: store,
        updateCamera: React.useRef(null),
        meshPropsRef: React.useRef(null),
      }}
    >
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
    somNumMotionGaussians: 0,
    near: 1.0,
    far: 100.0,
    depthTest: true,
    depthWrite: false,
    transparent: true,
    textureBuffer: null,
    somMotionCoeffsBuffer: null,
    somMotionBases: new Float32Array(30 * 4),
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

  // Shape of motion stuff.
  uniform uint somNumMotionGaussians;
  uniform usampler2D somMotionCoeffsBuffer;
  uniform vec4[30] somMotionBases;

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
    ivec2 texPos1 = ivec2((texStart + 1u) % uint(texSize.x), (texStart + 1u) / uint(texSize.x));


    // Get shape of motion transform.
    mat3 somRotation = mat3(1.0);
    vec3 somTranslation = vec3(0.0);
    float coeffs[10];
    if (sortedIndex < somNumMotionGaussians) {
      uvec4 coeffsVec0 = texelFetch(somMotionCoeffsBuffer, texPos0, 0);
      uvec4 coeffsVec1 = texelFetch(somMotionCoeffsBuffer, texPos1, 0);

      // Unpack first vec2 into array indices 0,1
      vec2 temp = unpackHalf2x16(coeffsVec0.x);
      coeffs[0] = temp.x;
      coeffs[1] = temp.y;

      // Unpack second vec2 into indices 2,3
      temp = unpackHalf2x16(coeffsVec0.y);
      coeffs[2] = temp.x;
      coeffs[3] = temp.y;

      // Indices 4,5
      temp = unpackHalf2x16(coeffsVec0.z);
      coeffs[4] = temp.x;
      coeffs[5] = temp.y;

      // Indices 6,7
      temp = unpackHalf2x16(coeffsVec0.w);
      coeffs[6] = temp.x;
      coeffs[7] = temp.y;

      // Final two coefficients 8,9
      temp = unpackHalf2x16(coeffsVec1.x);
      coeffs[8] = temp.x;
      coeffs[9] = temp.y;

      // uniform vec4[30] somMotionBases;
      // vec4: x y z _
      // vec4: rot0 rot1 rot2 _
      // vec4: rot3 rot4 rot5 _

      vec3 rot_x = vec3(0.0);
      vec3 rot_y = vec3(0.0);

      for (int i = 0; i < 10; i++) {
        somTranslation += vec3(somMotionBases[i * 3]) * coeffs[i];
        rot_x += vec3(somMotionBases[i * 3 + 1]) * coeffs[i];
        rot_y += vec3(somMotionBases[i * 3 + 2]) * coeffs[i];
      }

      // Use rot_x and rot_y as 6D continuous rotation representation.
      rot_x = normalize(rot_x);
      rot_y = normalize(rot_y - dot(rot_x, rot_y) * rot_x);
      vec3 rot_z = normalize(cross(rot_x, rot_y));

      somRotation = mat3(
        rot_x, rot_y, rot_z
      );
    }


    // Fetch from textures.
    uvec4 floatBufferData = texelFetch(textureBuffer, texPos0, 0);
    mat4 T_camera_group = getGroupTransform(floatBufferData.w);

    // Any early return will discard the fragment.
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec3 center = uintBitsToFloat(floatBufferData.xyz);

    //  Apply shape of motion transform.
    center = somRotation * center + somTranslation;

    vec4 c_cam = T_camera_group * vec4(center, 1);
    if (-c_cam.z < near || -c_cam.z > far)
      return;
    vec4 pos2d = projectionMatrix * c_cam;
    float clip = 1.1 * pos2d.w;
    if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip)
      return;

    // Read covariance terms.
    uvec4 intBufferData = texelFetch(textureBuffer, texPos1, 0);

    // Get covariance terms from int buffer.
    uint rgbaUint32 = intBufferData.w;
    vec2 triu01 = unpackHalf2x16(intBufferData.x);
    vec2 triu23 = unpackHalf2x16(intBufferData.y);
    vec2 triu45 = unpackHalf2x16(intBufferData.z);

    // Transition in.
    float startTime = 0.8 * float(sortedIndex) / float(numGaussians);
    float cov_scale = smoothstep(startTime, startTime + 0.2, transitionInState);

    // Do the actual splatting.
    mat3 cov3d = mat3(
        triu01.x, triu01.y, triu23.x,
        triu01.y, triu23.y, triu45.x,
        triu23.x, triu45.x, triu45.y
    );

    // Apply shape of motion to covariance.
    cov3d = somRotation * cov3d * transpose(somRotation);

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
    if (weightedDeterminant < 0.25)
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
    motionCoeffsBuffer: Uint32Array;
    numMotionGaussians: number;
  }
>(function SplatObject(
  { buffer, motionCoeffsBuffer, numMotionGaussians },
  ref,
) {
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const setBuffers = splatContext.useGaussianSplatStore(
    (state) => state.setBuffers,
  );
  const removeBuffers = splatContext.useGaussianSplatStore(
    (state) => state.removeBuffers,
  );
  const nodeRefFromId = splatContext.useGaussianSplatStore(
    (state) => state.nodeRefFromId,
  );
  const name = React.useMemo(() => uuidv4(), [buffer]);

  React.useEffect(() => {
    setBuffers(name, [buffer, motionCoeffsBuffer, numMotionGaussians]);
    return () => {
      removeBuffers(name);
      delete nodeRefFromId.current[name];
    };
  }, [buffer]);

  return (
    <group
      ref={(obj) => {
        // We'll (a) forward the ref and (b) store it in the splat rendering
        // state. The latter is used to update the sorter and shader.
        if (obj === null) return;
        if (ref !== null) {
          if ("current" in ref) {
            ref.current = obj;
          } else {
            ref(obj);
          }
        }
        nodeRefFromId.current[name] = obj;
      }}
    ></group>
  );
});

/** External interface. Component should be added to the root of canvas.  */
function SplatRenderer() {
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const groupBufferFromId = splatContext.useGaussianSplatStore(
    (state) => state.groupBufferFromId,
  );

  // Only mount implementation (which will load sort worker, etc) if there are
  // Gaussians to render.
  return Object.keys(groupBufferFromId).length > 0 ? (
    <SplatRendererImpl />
  ) : null;
}

function SplatRendererImpl() {
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const groupBufferFromId = splatContext.useGaussianSplatStore(
    (state) => state.groupBufferFromId,
  );
  const nodeRefFromId = splatContext.useGaussianSplatStore(
    (state) => state.nodeRefFromId,
  );

  // Consolidate Gaussian groups into a single buffer.
  const merged = mergeGaussianGroups(groupBufferFromId);
  const meshProps = useGaussianMeshProps(
    merged.gaussianBuffer,
    merged.coeffsBuffer!,
    merged.numGroups,
    merged.numMotionGaussians,
  );
  splatContext.meshPropsRef.current = meshProps;

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
      meshProps.coeffsTexture.needsUpdate = true;
      initializedBufferTexture = true;
    }
  };
  function postToWorker(message: SorterWorkerIncoming) {
    sortWorker.postMessage(message);
  }
  postToWorker({
    setBuffer: [merged.gaussianBuffer, merged.coeffsBuffer!],
    setGroupIndices: merged.groupIndices,
  });
  postToWorker({
    setSomMotionBases: splatContext.shapeOfMotionBasesRef.current,
  });
  splatContext.postToWorkerRef.current = postToWorker;

  // Cleanup.
  React.useEffect(() => {
    return () => {
      meshProps.textureBuffer.dispose();
      meshProps.coeffsTexture.dispose();
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

  // Make local sorter. This will be used for blocking sorts, eg for rendering
  // from virtual cameras.
  const SorterRef = React.useRef<any>(null);
  React.useEffect(() => {
    (async () => {
      SorterRef.current = new (await MakeSorterModulePromise()).Sorter(
        merged.gaussianBuffer,
        merged.coeffsBuffer,
        merged.groupIndices,
      );
    })();
  }, [merged.gaussianBuffer, merged.groupIndices]);

  const updateCamera = React.useCallback(
    function updateCamera(
      camera: THREE.PerspectiveCamera,
      width: number,
      height: number,
      blockingSort: boolean,
    ) {
      // Update camera parameter uniforms.
      const fovY = ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0;

      const aspect = width / height;
      const fovX = 2 * Math.atan(Math.tan(fovY / 2) * aspect);
      const fy = height / (2 * Math.tan(fovY / 2));
      const fx = width / (2 * Math.tan(fovX / 2));

      if (meshProps.material === undefined) return;

      const uniforms = meshProps.material.uniforms;
      uniforms.focal.value = [fx, fy];
      uniforms.near.value = camera.near;
      uniforms.far.value = camera.far;
      uniforms.viewport.value = [width, height];

      // Update group transforms.
      camera.updateMatrixWorld();
      const T_camera_world = camera.matrixWorldInverse;
      const groupVisibles: boolean[] = [];
      let visibilitiesChanged = false;
      for (const [groupIndex, name] of Object.keys(
        groupBufferFromId,
      ).entries()) {
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
        if (blockingSort && SorterRef.current !== null) {
          const sortedIndices = SorterRef.current.sort(
            Tz_camera_groups,
          ) as Uint32Array;
          meshProps.sortedIndexAttribute.set(sortedIndices);
          meshProps.sortedIndexAttribute.needsUpdate = true;
        } else {
          postToWorker({
            setTz_camera_groups: Tz_camera_groups,
          });
        }
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
    },
    [meshProps],
  );
  splatContext.updateCamera.current = updateCamera;

  useFrame((state, delta) => {
    const mesh = meshRef.current;
    if (
      mesh === null ||
      sortWorker === null ||
      meshProps.rowMajorT_camera_groups.length === 0
    )
      return;

    const uniforms = meshProps.material.uniforms;
    uniforms.transitionInState.value = Math.min(
      uniforms.transitionInState.value + delta * 2.0,
      1.0,
    );

    updateCamera(
      state.camera as THREE.PerspectiveCamera,
      state.viewport.dpr * state.size.width,
      state.viewport.dpr * state.size.height,
      false /* blockingSort */,
    );
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
  [name: string]: [Uint32Array, Uint32Array, number];
}) {
  // Create geometry. Each Gaussian will be rendered as a quad.
  let totalBufferLength = 0;
  for (const [buffer] of Object.values(groupBufferFromName)) {
    totalBufferLength += buffer.length;
  }
  const numGaussians = totalBufferLength / 8;
  const gaussianBuffer = new Uint32Array(totalBufferLength);
  const groupIndices = new Uint32Array(numGaussians);

  let coeffsBuffer = null;
  let numMotionGaussians = 0;

  let offset = 0;
  for (const [
    groupIndex,
    [groupBuffer, groupMotionCoeffsBuffer, groupNumMotionGaussians],
  ] of Object.values(groupBufferFromName).entries()) {
    // <HACKS> we're going to assume that there's only 1 group.
    if (Object.keys(groupBufferFromName).length !== 1) {
      console.error(
        "SHAPE OF MOTION HACKS: WE'RE GOING TO ASSUME ONLY  1 GROUP, BUT WE FOUND ",
        Object.keys(groupBufferFromName).length,
      );
    }
    coeffsBuffer = groupMotionCoeffsBuffer;
    numMotionGaussians = groupNumMotionGaussians;
    // </HACKS>

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
  return {
    numGaussians,
    coeffsBuffer,
    numMotionGaussians,
    gaussianBuffer,
    numGroups,
    groupIndices,
  };
}

/**Hook to generate properties for rendering Gaussians via a three.js mesh.*/
function useGaussianMeshProps(
  gaussianBuffer: Uint32Array,
  motionCoeffsBuffer: Uint32Array,
  numGroups: number,
  numMotionGaussians: number,
) {
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

  // Create texture buffers for group transforms.
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

  // Create texture buffers for shape of motion coefficients.
  const coeffsTextureWidth = Math.min(numGaussians * 2, maxTextureSize);
  const coeffsTextureHeight = Math.ceil(
    (numGaussians * 2) / coeffsTextureWidth,
  );
  const coeffsBufferPadded = new Uint32Array(
    coeffsTextureWidth * coeffsTextureHeight * 4,
  );
  console.log("widths", textureWidth, coeffsTextureWidth);
  console.log("heights", textureHeight, coeffsTextureHeight);
  coeffsBufferPadded.set(motionCoeffsBuffer);
  const coeffsTexture = new THREE.DataTexture(
    coeffsBufferPadded,
    coeffsTextureWidth,
    coeffsTextureHeight,
    THREE.RGBAIntegerFormat,
    THREE.UnsignedIntType,
  );
  coeffsTexture.internalFormat = "RGBA32UI";
  coeffsTexture.needsUpdate = true;

  const splatContext = React.useContext(GaussianSplatsContext)!;
  const bases = splatContext.shapeOfMotionBasesRef.current;
  const args: any = {
    textureBuffer: textureBuffer,
    textureT_camera_groups: textureT_camera_groups,
    numGaussians: 0,
    transitionInState: 0.0,
    // shape of motion stuff
    somNumMotionGaussians: numMotionGaussians,
    somMotionCoeffsBuffer: coeffsTexture,
    somMotionBases: bases,
  };
  const material = new GaussianSplatMaterial(args);
  // const bases = splatContext.shapeOfMotionBasesRef.current;
  splatContext.materialRef.current = material;

  return {
    geometry,
    material,
    textureBuffer,
    coeffsTexture,
    sortedIndexAttribute,
    textureT_camera_groups,
    rowMajorT_camera_groups,
  };
}
