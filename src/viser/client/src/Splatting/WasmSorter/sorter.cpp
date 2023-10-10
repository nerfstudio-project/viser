#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

struct Gaussian {
  float center[3];
  float rgb[3];
  float opacity;
  float cov_a[3];
  float cov_b[3];
};

struct Buffers {
  std::vector<float> centers;
  std::vector<float> rgbs;
  std::vector<float> opacities;
  std::vector<float> cov_a;
  std::vector<float> cov_b;
};

// Build a Float32Array view of a C++ vector.
emscripten::val getFloat32Array(const std::vector<float> &v) {
  return emscripten::val(emscripten::typed_memory_view(v.size(), &(v[0])));
}

class Sorter {
  // Properties for each unsorted Gaussian. A vector of structs (as opposed to a
  // struct of vectors) produces less fragmented reads. This can result in an
  // ~30% runtime improvement.
  std::vector<Gaussian> unsorted_gaussians;

  // Sorted buffers. The memory layout here is intended to match WebGL.
  Buffers sorted_buffers;

public:
  Sorter(const emscripten::val &centers, const emscripten::val &rgbs,
         const emscripten::val &opacities, const emscripten::val &cov_a,
         const emscripten::val &cov_b) {
    Buffers unsorted_buffers{
        emscripten::convertJSArrayToNumberVector<float>(centers),
        emscripten::convertJSArrayToNumberVector<float>(rgbs),
        emscripten::convertJSArrayToNumberVector<float>(opacities),
        emscripten::convertJSArrayToNumberVector<float>(cov_a),
        emscripten::convertJSArrayToNumberVector<float>(cov_b)};

    int num_gaussians = unsorted_buffers.centers.size() / 3;
    for (int i = 0; i < num_gaussians; i++) {
      unsorted_gaussians.push_back({
          {unsorted_buffers.centers[i * 3 + 0],
           unsorted_buffers.centers[i * 3 + 1],
           unsorted_buffers.centers[i * 3 + 2]},
          {unsorted_buffers.rgbs[i * 3 + 0], unsorted_buffers.rgbs[i * 3 + 1],
           unsorted_buffers.rgbs[i * 3 + 2]},
          unsorted_buffers.opacities[i],
          {unsorted_buffers.cov_a[i * 3 + 0], unsorted_buffers.cov_a[i * 3 + 1],
           unsorted_buffers.cov_a[i * 3 + 2]},
          {unsorted_buffers.cov_b[i * 3 + 0], unsorted_buffers.cov_b[i * 3 + 1],
           unsorted_buffers.cov_b[i * 3 + 2]},
      });
    }

    sorted_buffers = unsorted_buffers;
  };

  // Run sorting using the newest view projection matrix. Mutates internal
  // buffers.
  void sort(float view_proj_2, float view_proj_6, float view_proj_10) {
    const int num_gaussians = unsorted_gaussians.size();

    // We do a 16-bit counting sort. This is mostly translated from Kevin Kwok's
    // Javascript implementation:
    //     https://github.com/antimatter15/splat/blob/main/main.js
    std::vector<int> depths(num_gaussians);
    std::vector<int> counts0(256 * 256, 0);
    std::vector<int> starts0(256 * 256, 0);

    int min_depth;
    int max_depth;
    for (int i = 0; i < num_gaussians; i++) {
      const int depth = (((view_proj_2 * unsorted_gaussians[i].center[0] +
                           view_proj_6 * unsorted_gaussians[i].center[1] +
                           view_proj_10 * unsorted_gaussians[i].center[2]) *
                          4096.0));
      depths[i] = depth;

      if (i == 0 || depth < min_depth)
        min_depth = depth;
      if (i == 0 || depth > max_depth)
        max_depth = depth;
    }
    const float depth_inv = (256 * 256 - 1) / (max_depth - min_depth + 1e-5);
    for (int i = 0; i < num_gaussians; i++) {
      const int depth_bin = ((depths[i] - min_depth) * depth_inv);
      depths[i] = depth_bin;
      counts0[depth_bin]++;
    }
    for (int i = 1; i < 256 * 256; i++) {
      starts0[i] = starts0[i - 1] + counts0[i - 1];
    }

    std::vector<int> sorted_indices(num_gaussians);
    for (int i = 0; i < num_gaussians; i++)
      sorted_indices[starts0[depths[i]]++] = i;

    // Rearrange values in underlying buffers. This is the slowest part of the
    // sort.
    for (int i = 0; i < num_gaussians; i++) {
      const int j = sorted_indices[num_gaussians - i - 1];

      const Gaussian &gaussian = unsorted_gaussians[j];
      memcpy(&(sorted_buffers.centers[i * 3]), &gaussian.center, 4 * 3);
      memcpy(&(sorted_buffers.rgbs[i * 3]), &gaussian.rgb, 4 * 3);
      sorted_buffers.opacities[i] = gaussian.opacity;
      memcpy(&(sorted_buffers.cov_a[i * 3]), &gaussian.cov_a, 4 * 3);
      memcpy(&(sorted_buffers.cov_b[i * 3]), &gaussian.cov_b, 4 * 3);
    }
  }

  // Access outputs.
  emscripten::val getSortedCenters() {
    return getFloat32Array(sorted_buffers.centers);
  }
  emscripten::val getSortedRgbs() {
    return getFloat32Array(sorted_buffers.rgbs);
  }
  emscripten::val getSortedOpacities() {
    return getFloat32Array(sorted_buffers.opacities);
  }
  emscripten::val getSortedCovA() {
    return getFloat32Array(sorted_buffers.cov_a);
  }
  emscripten::val getSortedCovB() {
    return getFloat32Array(sorted_buffers.cov_b);
  }
};

EMSCRIPTEN_BINDINGS(c) {
  emscripten::class_<Sorter>("Sorter")
      .constructor<emscripten::val, emscripten::val, emscripten::val,
                   emscripten::val, emscripten::val>()
      .function("sort", &Sorter::sort)
      .function("getSortedCenters", &Sorter::getSortedCenters,
                emscripten::allow_raw_pointers())
      .function("getSortedRgbs", &Sorter::getSortedRgbs,
                emscripten::allow_raw_pointers())
      .function("getSortedOpacities", &Sorter::getSortedOpacities,
                emscripten::allow_raw_pointers())
      .function("getSortedCovA", &Sorter::getSortedCovA,
                emscripten::allow_raw_pointers())
      .function("getSortedCovB", &Sorter::getSortedCovB,
                emscripten::allow_raw_pointers());
};
