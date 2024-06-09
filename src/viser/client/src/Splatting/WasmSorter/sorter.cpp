#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

class Sorter {
    std::vector<std::array<float, 3>> unsorted_centers;
    std::vector<uint32_t> sorted_indices;

  public:
    Sorter(const emscripten::val &floatBuffer) {
        const std::vector<float> bufferVec =
            emscripten::convertJSArrayToNumberVector<float>(floatBuffer);
        const int num_gaussians = bufferVec.size() / 4;
        unsorted_centers.resize(num_gaussians);
        for (int i = 0; i < num_gaussians; i++) {
            unsorted_centers[i][0] = bufferVec[i * 4 + 0];
            unsorted_centers[i][1] = bufferVec[i * 4 + 1];
            unsorted_centers[i][2] = bufferVec[i * 4 + 2];
        }
    };

    // Run sorting using the newest view projection matrix. Mutates internal
    // buffers.
    emscripten::val
    sort(float view_proj_2, float view_proj_6, float view_proj_10) {
        const int num_gaussians = unsorted_centers.size();

        // We do a 16-bit counting sort. This is mostly translated from Kevin
        // Kwok's Javascript implementation:
        //     https://github.com/antimatter15/splat/blob/main/main.js
        std::vector<int> depths(num_gaussians);
        std::vector<int> counts0(256 * 256, 0);
        std::vector<int> starts0(256 * 256, 0);

        int min_depth;
        int max_depth;
        for (int i = 0; i < num_gaussians; i++) {
            const int depth =
                -(((view_proj_2 * unsorted_centers[i][0] +
                    view_proj_6 * unsorted_centers[i][1] +
                    view_proj_10 * unsorted_centers[i][2]) *
                   4096.0));
            depths[i] = depth;

            if (i == 0 || depth < min_depth)
                min_depth = depth;
            if (i == 0 || depth > max_depth)
                max_depth = depth;
        }
        const float depth_inv =
            (256 * 256 - 1) / (max_depth - min_depth + 1e-5);
        for (int i = 0; i < num_gaussians; i++) {
            const int depth_bin = ((depths[i] - min_depth) * depth_inv);
            depths[i] = depth_bin;
            counts0[depth_bin]++;
        }
        for (int i = 1; i < 256 * 256; i++) {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        // Update and return sorted indices.
        sorted_indices.resize(num_gaussians);
        for (int i = 0; i < num_gaussians; i++)
            sorted_indices[starts0[depths[i]]++] = i;

        return emscripten::val(emscripten::typed_memory_view(
            sorted_indices.size(), &(sorted_indices[0])
        ));
    }
};

EMSCRIPTEN_BINDINGS(c) {
    emscripten::class_<Sorter>("Sorter")
        .constructor<emscripten::val>()
        .function("sort", &Sorter::sort, emscripten::allow_raw_pointers());
};
