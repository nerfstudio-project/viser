#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

class Sorter {
    std::vector<std::array<float, 3>> unsorted_centers;
    std::vector<uint32_t> sorted_indices;
    float min_visible_depth;

  public:
    Sorter(const emscripten::val &buffer) {
        const std::vector<uint32_t> bufferVec =
            emscripten::convertJSArrayToNumberVector<uint32_t>(buffer);
        const int num_gaussians = bufferVec.size() / 8;
        unsorted_centers.resize(num_gaussians);
        for (int i = 0; i < num_gaussians; i++) {
            unsorted_centers[i][0] =
                reinterpret_cast<const float &>(bufferVec[i * 8 + 0]);
            unsorted_centers[i][1] =
                reinterpret_cast<const float &>(bufferVec[i * 8 + 1]);
            unsorted_centers[i][2] =
                reinterpret_cast<const float &>(bufferVec[i * 8 + 2]);
        }
    };

    // Run sorting using the newest view projection matrix. Mutates internal
    // buffers.
    emscripten::val sort(
        // column-major matrix ordering:
        // 0 4 8  12
        // 1 5 9  13
        // 2 6 10 14
        // 3 7 11 15
        const float view_proj_2,
        const float view_proj_6,
        const float view_proj_10,
        const float view_proj_14
    ) {
        const int num_gaussians = unsorted_centers.size();

        // We do a 16-bit counting sort. This is mostly translated from Kevin
        // Kwok's Javascript implementation:
        //     https://github.com/antimatter15/splat/blob/main/main.js
        //
        // Note: we want to sort from minimum Z (high depth) to maximum Z (low
        // depth).
        std::vector<int> gaussian_zs(num_gaussians);
        std::vector<int> counts0(256 * 256, 0);
        std::vector<int> starts0(256 * 256, 0);

        int min_z;
        int max_z;
        min_visible_depth = 100000.0;
        for (int i = 0; i < num_gaussians; i++) {
            const float cam_z = view_proj_2 * unsorted_centers[i][0] +
                                view_proj_6 * unsorted_centers[i][1] +
                                view_proj_10 * unsorted_centers[i][2] +
                                view_proj_14;

            // OpenGL camera convention: -Z is forward.
            const float depth = -cam_z;
            if (depth > 1e-4 && depth < min_visible_depth) {
                min_visible_depth = depth;
            }

            const int z_int = cam_z * 4096.0;
            gaussian_zs[i] = z_int;

            if (i == 0 || z_int < min_z)
                min_z = z_int;
            if (i == 0 || z_int > max_z)
                max_z = z_int;
        }

        const float z_inv = (256 * 256 - 1) / (max_z - min_z + 1e-5);
        for (int i = 0; i < num_gaussians; i++) {
            const int z_bin = ((gaussian_zs[i] - min_z) * z_inv);
            gaussian_zs[i] = z_bin;
            counts0[z_bin]++;
        }
        for (int i = 1; i < 256 * 256; i++) {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        // Update and return sorted indices.
        sorted_indices.resize(num_gaussians);
        for (int i = 0; i < num_gaussians; i++)
            sorted_indices[starts0[gaussian_zs[i]]++] = i;

        return emscripten::val(emscripten::typed_memory_view(
            sorted_indices.size(), &(sorted_indices[0])
        ));
    }

    float getMinDepth() { return min_visible_depth; }
};

EMSCRIPTEN_BINDINGS(c) {
    emscripten::class_<Sorter>("Sorter")
        .constructor<emscripten::val>()
        .function("sort", &Sorter::sort, emscripten::allow_raw_pointers())
        .function("getMinDepth", &Sorter::getMinDepth);
};
