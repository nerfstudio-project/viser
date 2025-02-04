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
import { useFrame } from "@react-three/fiber";
import { SorterWorkerIncoming } from "./SplatSortWorker";
import { v4 as uuidv4 } from "uuid";

import {
  GaussianSplatsContext,
  useGaussianMeshProps,
  useGaussianSplatStore,
} from "./GaussianSplatsHelpers";

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
export const SplatObject = React.forwardRef<
  THREE.Group,
  {
    buffer: Uint32Array;
  }
>(function SplatObject({ buffer }, ref) {
  const splatContext = React.useContext(GaussianSplatsContext)!;
  const setBuffer = splatContext.useGaussianSplatStore(
    (state) => state.setBuffer,
  );
  const removeBuffer = splatContext.useGaussianSplatStore(
    (state) => state.removeBuffer,
  );
  const nodeRefFromId = splatContext.useGaussianSplatStore(
    (state) => state.nodeRefFromId,
  );
  const name = React.useMemo(() => uuidv4(), [buffer]);

  React.useEffect(() => {
    setBuffer(name, buffer);
    return () => {
      removeBuffer(name);
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
    merged.numGroups,
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

  // Make local sorter. This will be used for blocking sorts, eg for rendering
  // from virtual cameras.
  const SorterRef = React.useRef<any>(null);
  React.useEffect(() => {
    (async () => {
      SorterRef.current = new (await MakeSorterModulePromise()).Sorter(
        merged.gaussianBuffer,
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
