/** Worker for sorting splats.
 */

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

{
  let sorter: any = null;
  let viewProj: number[] | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (sorter === null) {
      setTimeout(throttledSort, 1);
      return;
    }
    if (viewProj === null || sortRunning) return;

    sortRunning = true;
    const lastView = viewProj;
    const sortedIndices = sorter.sort(viewProj[2], viewProj[6], viewProj[10]);
    self.postMessage({
      sortedIndices: sortedIndices,
    });

    setTimeout(() => {
      sortRunning = false;
      if (lastView !== viewProj) {
        throttledSort();
      }
    }, 0);
  };

  const SorterModulePromise = MakeSorterModulePromise();

  self.onmessage = async (e) => {
    const data = e.data as
      | {
          setFloatBuffer: Float32Array;
        }
      | {
          setViewProj: number[];
        }
      | { close: true };

    if ("setFloatBuffer" in data) {
      // Instantiate sorter with buffers populated.
      sorter = new (await SorterModulePromise).Sorter(data.setFloatBuffer);
    } else if ("setViewProj" in data) {
      // Update view projection matrix.
      viewProj = data.setViewProj;
      throttledSort();
    } else if ("close" in data) {
      // Done!
      self.close();
    }
  };
}
