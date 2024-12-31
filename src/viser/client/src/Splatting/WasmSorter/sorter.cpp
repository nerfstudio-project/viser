#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <wasm_simd128.h>

#include <array>
#include <cstdint>
#include <vector>

#include <math.h>
#include <stdint.h>

v128_t cross_f32x4(v128_t a, v128_t b) {
    // For vectors a = (a1, a2, a3, a4) and b = (b1, b2, b3, b4)
    // Cross product = (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1, 0)

    // Shuffle components for cross product calculation
    v128_t a_yzx = wasm_i32x4_shuffle(a, a, 1, 2, 0, 3); // (a2, a3, a1, a4)
    v128_t b_yzx = wasm_i32x4_shuffle(b, b, 1, 2, 0, 3); // (b2, b3, b1, b4)
    v128_t a_zxy = wasm_i32x4_shuffle(a, a, 2, 0, 1, 3); // (a3, a1, a2, a4)
    v128_t b_zxy = wasm_i32x4_shuffle(b, b, 2, 0, 1, 3); // (b3, b1, b2, b4)

    // Multiply shuffled vectors
    v128_t mul1 = wasm_f32x4_mul(a_yzx, b_zxy); // (a2*b3, a3*b1, a1*b2, a4*b4)
    v128_t mul2 = wasm_f32x4_mul(a_zxy, b_yzx); // (a3*b2, a1*b3, a2*b1, a4*b4)

    // Subtract to get cross product
    v128_t result = wasm_f32x4_sub(mul1, mul2);

    // Zero out the last component
    result = wasm_f32x4_replace_lane(result, 3, 0.0f);

    return result;
}

// Convert a float16 bit pattern to float32
float float16_to_float32(uint16_t float16) {
    // Extract components
    uint32_t sign = (float16 >> 15) & 0x1;
    uint32_t exp = (float16 >> 10) & 0x1F;
    uint32_t frac = float16 & 0x3FF;

    // Handle special cases
    if (exp == 0) {
        if (frac == 0) {
            // Zero
            return sign ? -0.0f : 0.0f;
        } else {
            // Denormal
            float result = (float)frac * powf(2.0f, -24.0f);
            return sign ? -result : result;
        }
    } else if (exp == 31) {
        if (frac == 0) {
            // Infinity
            return sign ? -INFINITY : INFINITY;
        } else {
            // NaN
            return NAN;
        }
    }

    // Normal number
    // Convert to float32 format
    uint32_t f32 = (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13);
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

// Unpack two float16s from a uint32
void unpack_float16s(uint32_t packed, float *first, float *second) {
    uint16_t float16_1 = (packed >> 16) & 0xFFFF; // Extract upper 16 bits
    uint16_t float16_2 = packed & 0xFFFF;         // Extract lower 16 bits

    *first = float16_to_float32(float16_1);
    *second = float16_to_float32(float16_2);
}

/** SIMD dot product between two 4D vectors. */
__attribute__((always_inline)) inline float
dot_f32x4(const v128_t &a, const v128_t &b) {
    v128_t product = wasm_f32x4_mul(a, b);
    v128_t temp = wasm_f32x4_add(
        product, wasm_i32x4_shuffle(product, product, 1, 0, 3, 2)
    );
    v128_t tmp =
        wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 3, 0, 1));
    return wasm_f32x4_extract_lane(tmp, 0);
}

// Function to find the minimum value across a v128_t i32x4 vector.
__attribute__((always_inline)) inline int32_t min_i32x4(v128_t vector) {
    int32_t elem0 = wasm_i32x4_extract_lane(vector, 0);
    int32_t elem1 = wasm_i32x4_extract_lane(vector, 1);
    int32_t elem2 = wasm_i32x4_extract_lane(vector, 2);
    int32_t elem3 = wasm_i32x4_extract_lane(vector, 3);
    return std::min({elem0, elem1, elem2, elem3});
}

// Function to find the maximum value across a v128_t i32x4 vector and return it
// as a float.
__attribute__((always_inline)) inline int32_t max_i32x4(v128_t vector) {
    int32_t elem0 = wasm_i32x4_extract_lane(vector, 0);
    int32_t elem1 = wasm_i32x4_extract_lane(vector, 1);
    int32_t elem2 = wasm_i32x4_extract_lane(vector, 2);
    int32_t elem3 = wasm_i32x4_extract_lane(vector, 3);
    return std::max({elem0, elem1, elem2, elem3});
}

class Sorter {
    std::vector<v128_t> centers_homog; // Centers as homogeneous coordinates.
    std::vector<uint32_t> group_indices;
    std::vector<uint32_t> sorted_indices;
    std::vector<std::array<float, 10>> coeffs;

  public:
    Sorter(
        const emscripten::val &buffer,
        const emscripten::val &coeffsBuffer,
        const emscripten::val &group_indices_val
    ) {
        const std::vector<uint32_t> bufferVec =
            emscripten::convertJSArrayToNumberVector<uint32_t>(buffer);
        const float *floatBuffer =
            reinterpret_cast<const float *>(bufferVec.data());

        const std::vector<uint32_t> coeffsBufferVec =
            emscripten::convertJSArrayToNumberVector<uint32_t>(coeffsBuffer);

        const int32_t num_gaussians = bufferVec.size() / 8;
        sorted_indices.resize(num_gaussians);
        centers_homog.resize(num_gaussians);

        for (int32_t i = 0; i < num_gaussians; i++) {
            centers_homog[i] = wasm_f32x4_make(
                floatBuffer[i * 8 + 0],
                floatBuffer[i * 8 + 1],
                floatBuffer[i * 8 + 2],
                1.0
            );
        }

        int num_motion_gaussians = coeffsBufferVec.size() / 8;
        coeffs.resize(num_motion_gaussians);
        for (int32_t i = 0; i < num_motion_gaussians; i++) {
            for (int32_t j = 0; j < 5; j++) {
                float a, b;
                unpack_float16s(coeffsBufferVec[i * 8 + j], &a, &b);
                coeffs[i][j * 2 + 0] = b;
                coeffs[i][j * 2 + 1] = a;
            }
        }
        group_indices =
            emscripten::convertJSArrayToNumberVector<uint32_t>(group_indices_val
            );
    };

    // Run sorting using the newest view projection matrix. Mutates internal
    // buffers.
    emscripten::val sort(
        const emscripten::val &Tz_cam_groups_val,
        const emscripten::val &motion_bases
    ) {
        const auto motion_bases_buffer =
            emscripten::convertJSArrayToNumberVector<float>(motion_bases);

        std::vector centers_homog_transformed = centers_homog;
        const int32_t num_gaussians = centers_homog_transformed.size();

        for (int i = 0; i < coeffs.size(); i++) {
            v128_t trans = wasm_f32x4_splat(0.0f);
            v128_t col_x = wasm_f32x4_splat(0.0f);
            v128_t col_y = wasm_f32x4_splat(0.0f);
            for (int j = 0; j < 10; j++) {
                v128_t coeff = wasm_f32x4_splat(coeffs[i][j]);
                trans = wasm_f32x4_add(
                    trans,
                    wasm_f32x4_mul(
                        coeff, wasm_v128_load(&motion_bases_buffer[j * 4 * 3])
                    )
                );
                col_x = wasm_f32x4_add(
                    col_x,
                    wasm_f32x4_mul(
                        coeff,
                        wasm_v128_load(&motion_bases_buffer[j * 4 * 3 + 4])
                    )
                );
                col_y = wasm_f32x4_add(
                    col_y,
                    wasm_f32x4_mul(
                        coeff,
                        wasm_v128_load(&motion_bases_buffer[j * 4 * 3 + 8])
                    )
                );
            }

            // The last lanes should all be zero.
            col_x = wasm_f32x4_replace_lane(col_x, 3, 0.0);
            col_y = wasm_f32x4_replace_lane(col_y, 3, 0.0);

            // Normalize col_x.
            col_x = wasm_f32x4_div(
                col_x, wasm_f32x4_splat(sqrtf(dot_f32x4(col_x, col_x)))
            );
            // Gram-schmidt.
            col_y = wasm_f32x4_sub(
                col_y,
                wasm_f32x4_mul(col_x, wasm_f32x4_splat(dot_f32x4(col_x, col_y)))
            );
            // Normalize col_y.
            col_y = wasm_f32x4_div(
                col_y, wasm_f32x4_splat(sqrtf(dot_f32x4(col_y, col_y)))
            );
            v128_t col_z = cross_f32x4(col_x, col_y);

            v128_t row_x = wasm_f32x4_make(
                wasm_f32x4_extract_lane(col_x, 0),
                wasm_f32x4_extract_lane(col_y, 0),
                wasm_f32x4_extract_lane(col_z, 0),
                0.0
            );
            v128_t row_y = wasm_f32x4_make(
                wasm_f32x4_extract_lane(col_x, 1),
                wasm_f32x4_extract_lane(col_y, 1),
                wasm_f32x4_extract_lane(col_z, 1),
                0.0
            );
            v128_t row_z = wasm_f32x4_make(
                wasm_f32x4_extract_lane(col_x, 2),
                wasm_f32x4_extract_lane(col_y, 2),
                wasm_f32x4_extract_lane(col_z, 2),
                0.0
            );

            centers_homog_transformed[i] = wasm_f32x4_add(
                wasm_f32x4_make(
                    dot_f32x4(row_x, centers_homog_transformed[i]),
                    dot_f32x4(row_y, centers_homog_transformed[i]),
                    dot_f32x4(row_z, centers_homog_transformed[i]),
                    1.0
                ),
                trans
            );
        }

        const auto Tz_cam_groups_buffer =
            emscripten::convertJSArrayToNumberVector<float>(Tz_cam_groups_val);

        // We do a 16-bit counting sort. This is mostly translated from Kevin
        // Kwok's Javascript implementation:
        //     https://github.com/antimatter15/splat/blob/main/main.js
        //
        // Note: we want to sort from minimum Z (high depth) to maximum Z (low
        // depth).
        const int32_t padded_length = std::ceil(num_gaussians / 4.0);
        std::vector<v128_t> gaussian_zs(padded_length);
        std::array<int32_t, 256 * 256> counts0({0});
        std::array<int32_t, 256 * 256> starts0({0});

        const int32_t num_groups = Tz_cam_groups_buffer.size() / 4;
        std::vector<v128_t> Tz_cam_groups(num_groups);

        const v128_t row3 = wasm_f32x4_make(0.0, 0.0, 0.0, 1.0);
        for (int32_t i = 0; i < num_groups; i++) {
            Tz_cam_groups[i] = wasm_v128_load(&Tz_cam_groups_buffer[i * 4]);
        }

        v128_t min_z_i32x4;
        v128_t max_z_i32x4;
        const v128_t splat4096 = wasm_f32x4_splat(4096.0);
        for (int32_t i = 0; i < padded_length; i++) {
            // This should get inlined.
            int32_t gaussianIndex = i * 4;
            const float z0 = dot_f32x4(
                Tz_cam_groups[group_indices[gaussianIndex]],
                centers_homog_transformed[gaussianIndex]
            );
            gaussianIndex++;
            const float z1 = dot_f32x4(
                Tz_cam_groups[group_indices[gaussianIndex]],
                centers_homog_transformed[gaussianIndex]
            );
            gaussianIndex++;
            const float z2 = dot_f32x4(
                Tz_cam_groups[group_indices[gaussianIndex]],
                centers_homog_transformed[gaussianIndex]
            );
            gaussianIndex++;
            const float z3 = dot_f32x4(
                Tz_cam_groups[group_indices[gaussianIndex]],
                centers_homog_transformed[gaussianIndex]
            );
            const v128_t cam_z = wasm_f32x4_make(z0, z1, z2, z3);

            // OpenGL camera convention: -Z is forward.
            const v128_t depth = wasm_f32x4_neg(cam_z);
            const v128_t z_int =
                wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(cam_z, splat4096));
            gaussian_zs[i] = z_int;

            if (i == 0) {
                min_z_i32x4 = z_int;
                max_z_i32x4 = z_int;
            } else {
                // Currently, we incorrectly include padding elements in the
                // min/max.
                min_z_i32x4 = wasm_i32x4_min(min_z_i32x4, z_int);
                max_z_i32x4 = wasm_i32x4_max(max_z_i32x4, z_int);
            }
        }
        min_z_i32x4 = wasm_i32x4_splat(min_i32x4(min_z_i32x4));
        max_z_i32x4 = wasm_i32x4_splat(max_i32x4(max_z_i32x4));
        const v128_t z_inv = wasm_f32x4_div(
            wasm_f32x4_splat(256 * 256 - 1),
            wasm_f32x4_add(
                wasm_f32x4_convert_i32x4(
                    wasm_i32x4_sub(max_z_i32x4, min_z_i32x4)
                ),
                wasm_f32x4_splat(1e-5f)
            )
        );
        for (int32_t i = 0; i < padded_length; i++) {
            const v128_t z_bin = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(
                wasm_f32x4_convert_i32x4(
                    wasm_i32x4_sub(gaussian_zs[i], min_z_i32x4)
                ),
                z_inv
            ));
            gaussian_zs[i] = z_bin;
            counts0[wasm_i32x4_extract_lane(z_bin, 0)]++;
            if (i == padded_length - 1) {
                if (i * 4 + 1 < num_gaussians)
                    counts0[wasm_i32x4_extract_lane(z_bin, 1)]++;
                if (i * 4 + 2 < num_gaussians)
                    counts0[wasm_i32x4_extract_lane(z_bin, 2)]++;
                if (i * 4 + 3 < num_gaussians)
                    counts0[wasm_i32x4_extract_lane(z_bin, 3)]++;
            } else {
                counts0[wasm_i32x4_extract_lane(z_bin, 1)]++;
                counts0[wasm_i32x4_extract_lane(z_bin, 2)]++;
                counts0[wasm_i32x4_extract_lane(z_bin, 3)]++;
            }
        }
        for (int32_t i = 1; i < 256 * 256; i++) {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        // Update and return sorted indices.
        for (int32_t i = 0; i < num_gaussians; i++)
            sorted_indices[starts0[((int32_t *)&gaussian_zs[0])[i]]++] = i;
        return emscripten::val(emscripten::typed_memory_view(
            sorted_indices.size(), &(sorted_indices[0])
        ));
    }
};

EMSCRIPTEN_BINDINGS(c) {
    emscripten::class_<Sorter>("Sorter")
        .constructor<emscripten::val, emscripten::val, emscripten::val>()
        .function("sort", &Sorter::sort, emscripten::allow_raw_pointers());
};
