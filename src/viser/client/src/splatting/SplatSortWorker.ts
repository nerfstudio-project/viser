/** Worker for sorting splats.
 *
 * Adapted from Kevin Kwok:
 *     https://github.com/antimatter15/splat/blob/main/main.js
 */

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
  // Worker state.
  let buffers: GaussianBuffersSplitCov | null = null;
  let sortedBuffers: GaussianBuffersSplitCov | null = null;

  let viewProj: number[] | null = null;
  let depthList = new Int32Array();
  let sortedIndices: number[] = [];

  const runSort = (viewProj: number[] | null) => {
    if (buffers === null || sortedBuffers === null || viewProj === null) return;

    const numGaussians = buffers.centers.length / 3;

    // Create new buffers.
    if (sortedIndices.length !== numGaussians) {
      depthList = new Int32Array(numGaussians);
      sortedIndices = [...Array(numGaussians).keys()];
    }

    // Compute depth for each Gaussian.
    let maxDepth = -Infinity;
    let minDepth = Infinity;
    for (let i = 0; i < depthList.length; i++) {
      const depth =
        ((viewProj[2] * buffers.centers[i * 3 + 0] +
          viewProj[6] * buffers.centers[i * 3 + 1] +
          viewProj[10] * buffers.centers[i * 3 + 2]) *
          4096) |
        0;
      depthList[i] = depth;
      if (depth > maxDepth) maxDepth = depth;
      if (depth < minDepth) minDepth = depth;
    }

    // This is a 16 bit single-pass counting sort.
    const depthInv = (256 * 256) / (maxDepth - minDepth);
    const counts0 = new Uint32Array(256 * 256);
    for (let i = 0; i < numGaussians; i++) {
      depthList[i] = ((depthList[i] - minDepth) * depthInv) | 0;
      counts0[depthList[i]]++;
    }
    const starts0 = new Uint32Array(256 * 256);
    for (let i = 1; i < 256 * 256; i++)
      starts0[i] = starts0[i - 1] + counts0[i - 1];
    for (let i = 0; i < numGaussians; i++)
      sortedIndices[starts0[depthList[i]]++] = i;

    // Sort and post underlying buffers.
    for (let i = 0; i < sortedIndices.length; i++) {
      const j = sortedIndices[sortedIndices.length - i - 1];
      sortedBuffers.centers[i * 3 + 0] = buffers.centers[j * 3 + 0];
      sortedBuffers.centers[i * 3 + 1] = buffers.centers[j * 3 + 1];
      sortedBuffers.centers[i * 3 + 2] = buffers.centers[j * 3 + 2];

      sortedBuffers.rgbs[i * 3 + 0] = buffers.rgbs[j * 3 + 0];
      sortedBuffers.rgbs[i * 3 + 1] = buffers.rgbs[j * 3 + 1];
      sortedBuffers.rgbs[i * 3 + 2] = buffers.rgbs[j * 3 + 2];

      sortedBuffers.opacities[i] = buffers.opacities[j];

      sortedBuffers.covA[i * 3 + 0] = buffers.covA[j * 3 + 0];
      sortedBuffers.covA[i * 3 + 1] = buffers.covA[j * 3 + 1];
      sortedBuffers.covA[i * 3 + 2] = buffers.covA[j * 3 + 2];

      sortedBuffers.covB[i * 3 + 0] = buffers.covB[j * 3 + 0];
      sortedBuffers.covB[i * 3 + 1] = buffers.covB[j * 3 + 1];
      sortedBuffers.covB[i * 3 + 2] = buffers.covB[j * 3 + 2];
    }
    self.postMessage(sortedBuffers);
  };

  let sortRunning = false;
  const throttledSort = () => {
    if (sortRunning) return;

    sortRunning = true;
    const lastView = viewProj;
    runSort(lastView);
    setTimeout(() => {
      sortRunning = false;
      if (lastView !== viewProj) {
        throttledSort();
      }
    }, 0);
  };

  self.onmessage = (e) => {
    const data = e.data as
      | {
          setBuffers: GaussianBuffersSplitCov;
        }
      | {
          setViewProj: number[];
        }
      | { close: true };

    if ("setBuffers" in data) {
      buffers = data.setBuffers;
      sortedBuffers = {
        centers: new Float32Array(buffers.centers.length),
        rgbs: new Float32Array(buffers.rgbs.length),
        opacities: new Float32Array(buffers.opacities.length),
        covA: new Float32Array(buffers.covA.length),
        covB: new Float32Array(buffers.covB.length),
      };
    }

    if ("setViewProj" in data) {
      viewProj = data.setViewProj;
      throttledSort();
    }

    if ("close" in data) {
      self.close();
    }
  };
}
