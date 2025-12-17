#include "ccm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

ColorCorrectionMatrix::ColorCorrectionMatrix(const float matrix[3][3], int bit_depth) {
    max_pixel_value = (1 << bit_depth) - 1;

    std::cout << "--- initalize CCM ---" << std::endl;
    std::cout << "Target Bit Depth: " << bit_depth << " (Max Val: " << max_pixel_value << ")" << std::endl;
    
    // ========== 检查点 3: Scale 与 Shift 一致性验证 ==========
    std::cout << "Fixed-Point Configuration:" << std::endl;
    std::cout << "  Q_FACTOR_SHIFT: " << Q_FACTOR_SHIFT << std::endl;
    std::cout << "  Q_FACTOR_SCALE: " << Q_FACTOR_SCALE << " (should equal 1 << " << Q_FACTOR_SHIFT << " = " << (1 << Q_FACTOR_SHIFT) << ")" << std::endl;
    std::cout << "  ROUNDING_OFFSET: " << ROUNDING_OFFSET << std::endl;
    if (Q_FACTOR_SCALE != (1 << Q_FACTOR_SHIFT)) {
        std::cerr << "  ✗ ERROR: Q_FACTOR_SCALE mismatch! Scale=" << Q_FACTOR_SCALE 
                  << ", Expected=" << (1 << Q_FACTOR_SHIFT) << std::endl;
    } else {
        std::cout << "  ✓ Scale/Shift consistency verified." << std::endl;
    }
    std::cout << "Converted Matrix (Integer):" << std::endl;

    // convert float matrix to fixed point matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matrix_fixed[i][j] = static_cast<int>(std::round(matrix[i][j] * Q_FACTOR_SCALE));
            std::cout << matrix_fixed[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
}

void ColorCorrectionMatrix::process(const std::vector<uint16_t>& src, std::vector<uint16_t>& dst) {
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }

    // 调试：打印前3个像素的输入输出值
    bool debug_enabled = true;
    int debug_count = 0;
    const int debug_limit = 3;

    for (size_t i = 0; i < src.size(); i += 3) {
        int r_in = src[i];
        int g_in = src[i + 1];
        int b_in = src[i + 2];

        int r_out = (r_in * matrix_fixed[0][0] + g_in * matrix_fixed[0][1] + b_in * matrix_fixed[0][2]);
        int g_out = (r_in * matrix_fixed[1][0] + g_in * matrix_fixed[1][1] + b_in * matrix_fixed[1][2]);
        int b_out = (r_in * matrix_fixed[2][0] + g_in * matrix_fixed[2][1] + b_in * matrix_fixed[2][2]);
        
        uint16_t r_final = clamp_pixel((r_out + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT);
        uint16_t g_final = clamp_pixel((g_out + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT);
        uint16_t b_final = clamp_pixel((b_out + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT);
        
        dst[i] = r_final;
        dst[i + 1] = g_final;
        dst[i + 2] = b_final;

        // ========== 检查点 4: 调试输出 ==========
        if (debug_enabled && debug_count < debug_limit) {
            std::cout << "  [Debug Pixel " << debug_count << "] Input RGB(" 
                      << r_in << ", " << g_in << ", " << b_in << ") -> ";
            std::cout << "Output RGB(" << r_final << ", " << g_final << ", " << b_final << ")";
            std::cout << " [Raw: " << r_out << ", " << g_out << ", " << b_out << "]" << std::endl;
            debug_count++;
        }
    }
}

uint16_t ColorCorrectionMatrix::clamp_pixel(int value) {
    if (value < 0) return 0;
    if (value > max_pixel_value) return static_cast<uint16_t>(max_pixel_value);
    return static_cast<uint16_t>(value);
}
