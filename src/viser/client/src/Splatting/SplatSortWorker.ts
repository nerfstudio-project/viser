/** Worker for sorting splats.
 */

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

export type SorterWorkerIncoming =
  | {
      setBuffer: Uint32Array;
      setGroupIndices: Uint32Array;
    }
  | {
      setT_world_groups: Float32Array;
    }
  | {
      setT_camera_world: number[];
    }
  | { triggerSort: true }
  | { close: true };

{
  let sorter: any = null;
  let T_camera_world: number[] | null = null;
  let T_world_groups: Float32Array | null = null;
  let Tz_cam_groups: Float32Array | null = null;
  let groupIndices: Uint32Array | null = null;
  let sortedGroupIndices: Uint32Array | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (
      sorter === null ||
      T_camera_world === null ||
      T_world_groups === null ||
      Tz_cam_groups === null ||
      groupIndices === null ||
      sortedGroupIndices === null
    ) {
      setTimeout(throttledSort, 1);
      return;
    }
    if (sortRunning) return;

    // Compute Tz_cam_groups.
    //
    // This is equivalent to getting the third row of `T_cam_world @
    // T_world_group`; it's a projection from a 3D point to a camera-frame
    // depth value (Z coordinate).
    const numGroups = T_world_groups.length / 12;
    for (let i = 0; i < numGroups; i++) {
      Tz_cam_groups[i * 4 + 0] =
        T_camera_world[2] * T_world_groups[i * 12 + 0] +
        T_camera_world[6] * T_world_groups[i * 12 + 4] +
        T_camera_world[10] * T_world_groups[i * 12 + 8];
      Tz_cam_groups[i * 4 + 1] =
        T_camera_world[2] * T_world_groups[i * 12 + 1] +
        T_camera_world[6] * T_world_groups[i * 12 + 5] +
        T_camera_world[10] * T_world_groups[i * 12 + 9];
      Tz_cam_groups[i * 4 + 2] =
        T_camera_world[2] * T_world_groups[i * 12 + 2] +
        T_camera_world[6] * T_world_groups[i * 12 + 6] +
        T_camera_world[10] * T_world_groups[i * 12 + 10];
      Tz_cam_groups[i * 4 + 3] =
        T_camera_world[2] * T_world_groups[i * 12 + 3] +
        T_camera_world[6] * T_world_groups[i * 12 + 7] +
        T_camera_world[10] * T_world_groups[i * 12 + 11] +
        T_camera_world[14];
    }

    sortRunning = true;
    const lastView = T_camera_world;
    const sortedIndices = sorter.sort(Tz_cam_groups);

    if (numGroups >= 2) {
      // Multiple groups: we need to update the per-Gaussian group indices.
      for (const [index, sortedIndex] of sortedIndices.entries()) {
        sortedGroupIndices[index] = groupIndices[sortedIndex];
      }
      self.postMessage({
        sortedIndices: sortedIndices,
        sortedGroupIndices: sortedGroupIndices,
      });
    } else {
      self.postMessage({
        sortedIndices: sortedIndices,
      });
    }

    setTimeout(() => {
      sortRunning = false;
      if (lastView !== T_camera_world) {
        throttledSort();
      }
    }, 0);
  };

  const SorterModulePromise = MakeSorterModulePromise();

  self.onmessage = async (e) => {
    const data = e.data as SorterWorkerIncoming;
    if ("setBuffer" in data) {
      // Instantiate sorter with buffers populated.
      sorter = new (await SorterModulePromise).Sorter(
        data.setBuffer,
        data.setGroupIndices,
      );
      groupIndices = data.setGroupIndices;
      sortedGroupIndices = groupIndices.slice();
    } else if ("setT_world_groups" in data) {
      // Update object transforms.
      T_world_groups = data.setT_world_groups;
      Tz_cam_groups = new Float32Array((T_world_groups.length / 12) * 4);
    } else if ("setT_camera_world" in data) {
      // Update view projection matrix.
      T_camera_world = data.setT_camera_world;
    } else if ("triggerSort" in data) {
      throttledSort();
    } else if ("close" in data) {
      // Done!
      self.close();
    }
  };
}
