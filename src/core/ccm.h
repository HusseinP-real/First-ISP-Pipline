#ifndef CCM_H
#define CCM_H

#include <vector>
#include <cstdint>

class ColorCorrectionMatrix {
private:
    static const int Q_FACTOR_SHIFT = 10;
    // 1024 - Fixed-point scale factor (2^10)
    static const int Q_FACTOR_SCALE = 1 << Q_FACTOR_SHIFT;
    // 512 - Rounding offset for proper rounding (half of Q_FACTOR_SCALE)
    // This ensures (value + ROUNDING_OFFSET) >> Q_FACTOR_SHIFT performs true rounding
    // instead of truncation, preventing darkening and loss of detail in shadows
    static const int ROUNDING_OFFSET = 1 << (Q_FACTOR_SHIFT - 1);

    // store the ccm matrix
    int matrix_fixed[3][3];

    int max_pixel_value;

    // Helper function to clamp pixel values
    inline uint16_t clamp_pixel(int value);

public:
    // Constructor: takes a 3x3 float matrix and bit depth (default 16)
    ColorCorrectionMatrix(const float matrix[3][3], int bit_depth = 16);

    // Process function: applies color correction matrix to input pixels
    // Input: src - source pixel data (RGB interleaved, 3 values per pixel)
    // Output: dst - destination pixel data (will be resized if needed)
    void process(const std::vector<uint16_t>& src, std::vector<uint16_t>& dst);
};

#endif // CCM_H

