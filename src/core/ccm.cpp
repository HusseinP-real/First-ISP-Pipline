#include "ccm.h"
#include <cmath>
#include <cstdint>

ColorCorrectionMatrix::ColorCorrectionMatrix(const float matrix[3][3], int bit_depth) {
    max_pixel_value = (1 << bit_depth) - 1;

    // convert float matrix to fixed point matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matrix_fixed[i][j] = static_cast<int>(std::round(matrix[i][j] * Q_FACTOR_SCALE));
        }
    }
}

void ColorCorrectionMatrix::process(const std::vector<uint16_t>& src, std::vector<uint16_t>& dst) {
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }

    for (size_t i = 0; i < src.size(); i += 3) {
        int r_in = src[i];
        int g_in = src[i + 1];
        int b_in = src[i + 2];

        // Use int64_t to prevent overflow in intermediate calculations
        // For 14-bit input (16383) with 12-bit scale (4096), max value ≈ 2×10^8
        // Using int64_t ensures safety even for extreme highlights
        int64_t r_out = (static_cast<int64_t>(r_in) * matrix_fixed[0][0] + 
                         static_cast<int64_t>(g_in) * matrix_fixed[0][1] + 
                         static_cast<int64_t>(b_in) * matrix_fixed[0][2]);
        int64_t g_out = (static_cast<int64_t>(r_in) * matrix_fixed[1][0] + 
                         static_cast<int64_t>(g_in) * matrix_fixed[1][1] + 
                         static_cast<int64_t>(b_in) * matrix_fixed[1][2]);
        int64_t b_out = (static_cast<int64_t>(r_in) * matrix_fixed[2][0] + 
                         static_cast<int64_t>(g_in) * matrix_fixed[2][1] + 
                         static_cast<int64_t>(b_in) * matrix_fixed[2][2]);
        
        uint16_t r_final = clamp_pixel(static_cast<int>((r_out + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT));
        uint16_t g_final = clamp_pixel(static_cast<int>((g_out + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT));
        uint16_t b_final = clamp_pixel(static_cast<int>((b_out + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT));
        
        dst[i] = r_final;
        dst[i + 1] = g_final;
        dst[i + 2] = b_final;
    }
}

uint16_t ColorCorrectionMatrix::clamp_pixel(int value) {
    if (value < 0) return 0;
    if (value > max_pixel_value) return static_cast<uint16_t>(max_pixel_value);
    return static_cast<uint16_t>(value);
}
