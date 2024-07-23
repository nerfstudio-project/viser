/** Worker for sorting splats.
 */

import MakeSorterModulePromise from "./WasmSorter/Sorter.mjs";

{
  let sorter: any = null;
  let T_camera_world: number[] | null = null;
  let T_world_objs: Float32Array | null = null;
  let sortRunning = false;
  const throttledSort = () => {
    if (sorter === null) {
      setTimeout(throttledSort, 1);
      return;
    }
    if (T_camera_world === null || sortRunning) return;

    sortRunning = true;
    const lastView = T_camera_world;
    const sortedIndices = sorter.sort(
      T_camera_world[2],
      T_camera_world[6],
      T_camera_world[10],
      T_camera_world[14],
      T_world_objs,
    );
    self.postMessage({
      sortedIndices: sortedIndices,
    });

    setTimeout(() => {
      sortRunning = false;
      if (lastView !== T_camera_world) {
        throttledSort();
      }
    }, 0);
  };

  const SorterModulePromise = MakeSorterModulePromise();

  self.onmessage = async (e) => {
    const data = e.data as
      | {
          setBuffer: Float32Array;
          setGroupIndices: Uint32Array;
        }
      | {
          setT_world_objs: Float32Array;
        }
      | {
          setT_camera_world: number[];
        }
      | { close: true };

    if ("setBuffer" in data) {
      // Instantiate sorter with buffers populated.
      sorter = new (await SorterModulePromise).Sorter(
        data.setBuffer,
        data.setGroupIndices,
      );
    } else if ("setT_world_objs" in data) {
      // Update object transforms.
      T_world_objs = data.setT_world_objs;
    } else if ("setT_camera_world" in data) {
      // Update view projection matrix.
      T_camera_world = data.setT_camera_world;
      throttledSort();
    } else if ("close" in data) {
      // Done!
      self.close();
    }
  };
}
