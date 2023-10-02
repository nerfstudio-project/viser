/** Worker for sorting splats.
 *
 * Adapted from Kevin Kwok:
 *     https://github.com/antimatter15/splat/blob/main/main.js
 */

{
  // Worker state.
  let buffers: {
    // (N, 3)
    centers: Float32Array;
    // (N, 3)
    rgbs: Float32Array;
    // (N, 1)
    opacities: Uint8Array;
    // (N, 3, 3)
    covariancesTriu: Float32Array;
  } | null = null;

  let viewProj: number[] | null = null;
  let sortedBuffers = {
    centers: new Float32Array(),
    rgbs: new Float32Array(),
    opacities: new Float32Array(),
    covA: new Float32Array(),
    covB: new Float32Array(),
  };
  let depthList = new Int32Array();
  let sortedIndices: number[] = [];

  const runSort = (viewProj: number[] | null) => {
    if (buffers === null || viewProj === null) return;

    const numGaussians = buffers.centers.length / 3;

    // Create new buffers.
    if (sortedBuffers.centers.length !== numGaussians * 3) {
      sortedBuffers = {
        centers: new Float32Array(numGaussians * 3),
        rgbs: new Float32Array(numGaussians * 3),
        opacities: new Float32Array(numGaussians * 1),
        covA: new Float32Array(numGaussians * 3),
        covB: new Float32Array(numGaussians * 3),
      };
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
    for (let j = 0; j < numGaussians; j++) {
      const i = sortedIndices[numGaussians - j - 1];

      sortedBuffers.centers[j * 3 + 0] = buffers.centers[i * 3 + 0];
      sortedBuffers.centers[j * 3 + 1] = buffers.centers[i * 3 + 1];
      sortedBuffers.centers[j * 3 + 2] = buffers.centers[i * 3 + 2];

      sortedBuffers.rgbs[j * 3 + 0] = buffers.rgbs[i * 3 + 0];
      sortedBuffers.rgbs[j * 3 + 1] = buffers.rgbs[i * 3 + 1];
      sortedBuffers.rgbs[j * 3 + 2] = buffers.rgbs[i * 3 + 2];

      sortedBuffers.opacities[j * 1 + 0] = buffers.opacities[i * 1 + 0] / 255.0;

      sortedBuffers.covA[3 * j + 0] = buffers.covariancesTriu[i * 6 + 0];
      sortedBuffers.covA[3 * j + 1] = buffers.covariancesTriu[i * 6 + 1];
      sortedBuffers.covA[3 * j + 2] = buffers.covariancesTriu[i * 6 + 2];
      sortedBuffers.covB[3 * j + 0] = buffers.covariancesTriu[i * 6 + 3];
      sortedBuffers.covB[3 * j + 1] = buffers.covariancesTriu[i * 6 + 4];
      sortedBuffers.covB[3 * j + 2] = buffers.covariancesTriu[i * 6 + 5];
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
          setBuffers: NonNullable<typeof buffers>;
        }
      | {
          setViewProj: number[];
        }
      | { close: true };

    if ("setBuffers" in data) {
      buffers = data.setBuffers;
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
