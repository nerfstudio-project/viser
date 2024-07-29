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
  | { close: true };

{
  let sorter: any = null;
  let Tz_camera_groups: Float32Array | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (sorter === null || Tz_camera_groups === null) {
      setTimeout(throttledSort, 1);
      return;
    }
    if (sortRunning) return;

    sortRunning = true;
    const lastView = Tz_camera_groups;

    // Important: we clone the output so we can transfer the buffer to the main
    // thread. Compared to relying on postMessage for copying, this reduces
    // backlog artifacts.
    const sortedIndices = (
      sorter.sort(Tz_camera_groups) as Uint32Array
    ).slice();

    // @ts-ignore
    self.postMessage({ sortedIndices: sortedIndices }, [sortedIndices.buffer]);

    setTimeout(() => {
      sortRunning = false;
      if (Tz_camera_groups === null) return;
      if (
        !lastView.every(
          // Cast is needed because of closure...
          (val, i) => val === (Tz_camera_groups as Float32Array)[i],
        )
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
    } else if ("setTz_camera_groups" in data) {
      // Update object transforms.
      Tz_camera_groups = data.setTz_camera_groups;
      throttledSort();
    } else if ("close" in data) {
      // Done!
      self.close();
    }
  };
}
