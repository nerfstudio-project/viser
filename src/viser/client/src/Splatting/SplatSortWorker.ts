/** Worker for sorting splats.
 */

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

{
  let sorter: any = null;
  let T_camera_obj: number[] | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (sorter === null) {
      setTimeout(throttledSort, 1);
      return;
    }
    if (T_camera_obj === null || sortRunning) return;

    sortRunning = true;
    const lastView = T_camera_obj;
    const sortedIndices = sorter.sort(
      T_camera_obj[2],
      T_camera_obj[6],
      T_camera_obj[10],
      T_camera_obj[14],
    );
    self.postMessage({
      sortedIndices: sortedIndices,
      minDepth: sorter.getMinDepth(),
    });

    setTimeout(() => {
      sortRunning = false;
      if (lastView !== T_camera_obj) {
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
          setT_camera_obj: number[];
        }
      | { close: true };

    if ("setFloatBuffer" in data) {
      // Instantiate sorter with buffers populated.
      sorter = new (await SorterModulePromise).Sorter(data.setFloatBuffer);
    } else if ("setT_camera_obj" in data) {
      // Update view projection matrix.
      T_camera_obj = data.setT_camera_obj;
      throttledSort();
    } else if ("close" in data) {
      // Done!
      self.close();
    }
  };
}
