/** Worker for sorting splats.
 */

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

export type GaussianBuffersSplitCov = {
  // (N, 3)
  centers: Float32Array;
  // (N, 3)
  rgbs: Float32Array;
  // (N, 1)
  opacities: Float32Array;
  // (N, 3)
  covA: Float32Array;
  // (N, 3)
  covB: Float32Array;
};

{
  let sorter: any = null;
  let viewProj: number[] | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (sorter === null || viewProj === null || sortRunning) return;

    sortRunning = true;
    const lastView = viewProj;
    sorter.sort(viewProj[2], viewProj[6], viewProj[10]);
    self.postMessage({
      centers: sorter.getSortedCenters(),
      rgbs: sorter.getSortedRgbs(),
      opacities: sorter.getSortedOpacities(),
      covA: sorter.getSortedCovA(),
      covB: sorter.getSortedCovB(),
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
          setBuffers: GaussianBuffersSplitCov;
        }
      | {
          setViewProj: number[];
        }
      | { close: true };

    if ("setBuffers" in data) {
      // Instantiate sorter with buffers populated.
      const buffers = data.setBuffers as GaussianBuffersSplitCov;
      sorter = new (await SorterModulePromise).Sorter(
        buffers.centers,
        buffers.rgbs,
        buffers.opacities,
        buffers.covA,
        buffers.covB,
      );
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
