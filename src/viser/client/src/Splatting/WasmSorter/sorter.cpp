#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <wasm_simd128.h>

#include <emscripten/bind.h>
#include <emscripten/val.h>

class Sorter {
    std::vector<std::array<float, 4>> unsorted_centers;
    std::vector<uint32_t> group_indices;

  public:
    Sorter(
        const emscripten::val &buffer, const emscripten::val &group_indices_val
    ) {
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
            unsorted_centers[i][3] = 1.0;
        }
        group_indices =
            emscripten::convertJSArrayToNumberVector<uint32_t>(group_indices_val
            );
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
        const float view_proj_14,
        const emscripten::val &T_world_groups_val
    ) {
        const auto T_world_groups =
            emscripten::convertJSArrayToNumberVector<float>(T_world_groups_val);
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

        // Put view_proj into a wasm v128 register.
        v128_t view_proj = wasm_f32x4_make(
            view_proj_2, view_proj_6, view_proj_10, view_proj_14
        );

        int min_z;
        int max_z;
        for (int i = 0; i < num_gaussians; i++) {
            const uint32_t group_index = group_indices[i];

            // This buffer is row-major.
            v128_t row0 = wasm_v128_load(&T_world_groups[group_index * 12]);
            v128_t row1 = wasm_v128_load(&T_world_groups[group_index * 12] + 4);
            v128_t row2 = wasm_v128_load(&T_world_groups[group_index * 12] + 8);
            v128_t point = wasm_v128_load(&unsorted_centers[i][0]);

            const auto dot_product = [](v128_t a, v128_t b) -> float {
                v128_t product = wasm_f32x4_mul(a, b);
                v128_t temp = wasm_f32x4_add(
                    product, wasm_i32x4_shuffle(product, product, 1, 0, 3, 2)
                );
                temp = wasm_f32x4_add(
                    temp, wasm_i32x4_shuffle(temp, temp, 2, 3, 0, 1)
                );
                return wasm_f32x4_extract_lane(temp, 0);
            };

            const float world_x = dot_product(row0, point);
            const float world_y = dot_product(row1, point);
            const float world_z = dot_product(row2, point);
            const float cam_z = dot_product(
                view_proj, wasm_f32x4_make(world_x, world_y, world_z, 1.0)
            );

            // OpenGL camera convention: -Z is forward.
            const float depth = -cam_z;
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
        auto sorted_indices = std::vector<uint32_t>(num_gaussians);
        for (int i = 0; i < num_gaussians; i++)
            sorted_indices[starts0[gaussian_zs[i]]++] = i;

        return emscripten::val(emscripten::typed_memory_view(
            sorted_indices.size(), &(sorted_indices[0])
        ));
    }
};

EMSCRIPTEN_BINDINGS(c) {
    emscripten::class_<Sorter>("Sorter")
        .constructor<emscripten::val, emscripten::val>()
        .function("sort", &Sorter::sort, emscripten::allow_raw_pointers());
};
