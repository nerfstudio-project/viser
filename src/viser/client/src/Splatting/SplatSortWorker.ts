/** Worker for sorting splats.
 */

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

export type SorterWorkerIncoming =
  | {
      setBuffer: Uint32Array;
      setGroupIndices: Uint32Array;
    }
  | {
      setTz_camera_groups: Float32Array;
    }
  | { triggerSort: true }
  | { close: true };

{
  let sorter: any = null;
  let Tz_camera_groups: Float32Array | null = null;
  let groupIndices: Uint32Array | null = null;
  let sortedGroupIndices: Uint32Array | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (
      sorter === null ||
      Tz_camera_groups === null ||
      groupIndices === null ||
      sortedGroupIndices === null
    ) {
      setTimeout(throttledSort, 1);
      return;
    }
    if (sortRunning) return;

    sortRunning = true;
    const lastView = Tz_camera_groups;
    const sortedIndices = sorter.sort(Tz_camera_groups);

    const numGroups = Tz_camera_groups.length / 4;
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
      if (
        Tz_camera_groups !== null &&
        !lastView.every((val, i) => val == Tz_camera_groups[i])
      ) {
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
    } else if ("setTz_camera_groups" in data) {
      // Update object transforms.
      Tz_camera_groups = data.setTz_camera_groups;
    } else if ("triggerSort" in data) {
      throttledSort();
    } else if ("close" in data) {
      // Done!
      self.close();
    }
  };
}
