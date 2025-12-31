/**
 * RCD (Ratio-based Color Difference) Demosaic Algorithm
 * 
 * Optimized implementation with:
 * - Correct handling of all Bayer patterns (RGGB, BGGR, GBRG, GRBG)
 * - Pre-padded buffers to eliminate boundary checks in hot loops
 * - OpenMP parallelization for row-level processing
 * - AVX2/SSE SIMD for gradient computation (optional, compile-time)
 * - Branch-free gradient direction selection using weighted average
 * - Cache-friendly memory access patterns
 * 
 * Author: ISP Algorithm Engineer
 * Target: 500x512, 16-bit, BGGR RAW
 */

#include "rcd.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// Check for SIMD support
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define USE_SSE4 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#endif

namespace {

// ============================================================================
// Bayer Pattern Configuration
// ============================================================================

/**
 * Get the starting offsets for each Bayer pattern
 * These offsets define where Red is located in the 2x2 Bayer cell
 */
inline void getPatternOffset(bayerPattern pattern, int& red_row_offset, int& red_col_offset) {
    switch (pattern) {
        case RGGB: red_row_offset = 0; red_col_offset = 0; break;  // R at (0,0)
        case BGGR: red_row_offset = 1; red_col_offset = 1; break;  // R at (1,1)
        case GBRG: red_row_offset = 1; red_col_offset = 0; break;  // R at (0,1)
        case GRBG: red_row_offset = 0; red_col_offset = 1; break;  // R at (1,0)
    }
}

/**
 * Pixel type enumeration for clearer logic
 */
enum class PixelType { Red, GreenOnRed, GreenOnBlue, Blue };

/**
 * Determine pixel type at (x, y) for given Bayer pattern
 */
inline PixelType getPixelType(int x, int y, int red_row_off, int red_col_off) {
    bool is_red_row = ((y + red_row_off) & 1) == 0;
    bool is_red_col = ((x + red_col_off) & 1) == 0;
    
    if (is_red_row && is_red_col) return PixelType::Red;
    if (!is_red_row && !is_red_col) return PixelType::Blue;
    if (is_red_row) return PixelType::GreenOnRed;  // Green in Red row
    return PixelType::GreenOnBlue;                  // Green in Blue row
}

/**
 * Check if pixel at (x, y) is a Green pixel
 */
inline bool isGreen(int x, int y, int red_row_off, int red_col_off) {
    bool is_red_row = ((y + red_row_off) & 1) == 0;
    bool is_red_col = ((x + red_col_off) & 1) == 0;
    return is_red_row != is_red_col;  // XOR: Green when row and col parity differ
}

// ============================================================================
// Padded Buffer for Boundary-Free Access
// ============================================================================

/**
 * Padded image buffer that allows accessing pixels at offset [-PAD, width+PAD)
 * without boundary checks. Uses reflect padding (mirror at edges).
 */
template<typename T, int PAD = 4>
class PaddedBuffer {
public:
    int width, height;
    int stride;  // padded width
    std::vector<T> data;

    PaddedBuffer() : width(0), height(0), stride(0) {}

    PaddedBuffer(int w, int h) : width(w), height(h) {
        stride = w + 2 * PAD;
        data.resize(stride * (h + 2 * PAD), T(0));
    }

    // Initialize from cv::Mat with reflect padding
    void initFromMat(const cv::Mat& src) {
        CV_Assert(src.channels() == 1);
        width = src.cols;
        height = src.rows;
        stride = width + 2 * PAD;
        data.resize(stride * (height + 2 * PAD));

        // Copy center region
        for (int y = 0; y < height; ++y) {
            const T* srcRow = src.ptr<T>(y);
            T* dstRow = row(y);
            std::copy(srcRow, srcRow + width, dstRow);
        }

        // Reflect padding: top and bottom
        for (int py = 0; py < PAD; ++py) {
            // Top padding: reflect row py -> row (PAD - 1 - py) mirrored
            T* topPad = data.data() + py * stride + PAD;
            const T* srcTop = row(PAD - 1 - py < height ? PAD - 1 - py : 0);
            std::copy(srcTop, srcTop + width, topPad);

            // Bottom padding
            T* botPad = data.data() + (height + PAD + py) * stride + PAD;
            int srcY = height - 1 - py;
            if (srcY < 0) srcY = 0;
            const T* srcBot = row(srcY);
            std::copy(srcBot, srcBot + width, botPad);
        }

        // Reflect padding: left and right (including corners)
        for (int y = -PAD; y < height + PAD; ++y) {
            T* r = row(y);
            for (int px = 0; px < PAD; ++px) {
                r[-PAD + px] = r[PAD - 1 - px < width ? PAD - 1 - px : 0];
                r[width + px] = r[width - 1 - px >= 0 ? width - 1 - px : 0];
            }
        }
    }

    // Initialize empty buffer
    void init(int w, int h) {
        width = w;
        height = h;
        stride = w + 2 * PAD;
        data.resize(stride * (h + 2 * PAD), T(0));
    }

    // Get pointer to row y (supports negative y for padding)
    inline T* row(int y) {
        return data.data() + (y + PAD) * stride + PAD;
    }

    inline const T* row(int y) const {
        return data.data() + (y + PAD) * stride + PAD;
    }

    // Direct access at (x, y) - no bounds checking needed for |offset| < PAD
    inline T& at(int x, int y) {
        return row(y)[x];
    }

    inline const T& at(int x, int y) const {
        return row(y)[x];
    }

    // Update padding after modifications (for intermediate buffers)
    void updatePadding() {
        // Reflect padding: top and bottom
        for (int py = 0; py < PAD; ++py) {
            T* topPad = data.data() + py * stride + PAD;
            const T* srcTop = row(std::min(PAD - 1 - py, height - 1));
            std::copy(srcTop, srcTop + width, topPad);

            T* botPad = data.data() + (height + PAD + py) * stride + PAD;
            const T* srcBot = row(std::max(0, height - 1 - py));
            std::copy(srcBot, srcBot + width, botPad);
        }

        // Reflect padding: left and right
        for (int y = -PAD; y < height + PAD; ++y) {
            T* r = row(y);
            for (int px = 0; px < PAD; ++px) {
                r[-PAD + px] = r[std::min(PAD - 1 - px, width - 1)];
                r[width + px] = r[std::max(0, width - 1 - px)];
            }
        }
    }
};

// ============================================================================
// SIMD Helper Functions
// ============================================================================

#if USE_AVX2
/**
 * Compute horizontal gradient for 8 consecutive pixels using AVX2
 * grad_h = |p[x-1] - p[x+1]| + |2*p[x] - p[x-2] - p[x+2]|
 */
inline __m256 computeGradH_AVX2(const float* p) {
    __m256 center = _mm256_loadu_ps(p);
    __m256 left1 = _mm256_loadu_ps(p - 1);
    __m256 right1 = _mm256_loadu_ps(p + 1);
    __m256 left2 = _mm256_loadu_ps(p - 2);
    __m256 right2 = _mm256_loadu_ps(p + 2);
    
    // |left1 - right1|
    __m256 diff1 = _mm256_sub_ps(left1, right1);
    __m256 abs1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff1);
    
    // |2*center - left2 - right2|
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 sum2 = _mm256_add_ps(left2, right2);
    __m256 diff2 = _mm256_sub_ps(_mm256_mul_ps(two, center), sum2);
    __m256 abs2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff2);
    
    return _mm256_add_ps(abs1, abs2);
}

/**
 * Compute vertical gradient for 8 consecutive pixels using AVX2
 * Requires pointers to rows: p_m2, p_m1, p, p_p1, p_p2
 */
inline __m256 computeGradV_AVX2(const float* p_m2, const float* p_m1, 
                                  const float* p, const float* p_p1, 
                                  const float* p_p2) {
    __m256 center = _mm256_loadu_ps(p);
    __m256 up1 = _mm256_loadu_ps(p_m1);
    __m256 down1 = _mm256_loadu_ps(p_p1);
    __m256 up2 = _mm256_loadu_ps(p_m2);
    __m256 down2 = _mm256_loadu_ps(p_p2);
    
    // |up1 - down1|
    __m256 diff1 = _mm256_sub_ps(up1, down1);
    __m256 abs1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff1);
    
    // |2*center - up2 - down2|
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 sum2 = _mm256_add_ps(up2, down2);
    __m256 diff2 = _mm256_sub_ps(_mm256_mul_ps(two, center), sum2);
    __m256 abs2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff2);
    
    return _mm256_add_ps(abs1, abs2);
}
#endif

// ============================================================================
// Scalar Gradient Functions (Fallback)
// ============================================================================

inline float computeGradH_Scalar(const float* p, int x) {
    float center = p[x];
    return std::abs(p[x - 1] - p[x + 1]) + 
           std::abs(2.0f * center - p[x - 2] - p[x + 2]);
}

inline float computeGradV_Scalar(const float* p_m2, const float* p_m1,
                                   const float* p, const float* p_p1,
                                   const float* p_p2, int x) {
    float center = p[x];
    return std::abs(p_m1[x] - p_p1[x]) + 
           std::abs(2.0f * center - p_m2[x] - p_p2[x]);
}

} // anonymous namespace

// ============================================================================
// Main RCD Demosaic Implementation
// ============================================================================

void demosiacRCD(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    CV_Assert(!raw.empty());
    CV_Assert(raw.channels() == 1);
    CV_Assert(raw.depth() == CV_16U || raw.depth() == CV_8U);

    const int width = raw.cols;
    const int height = raw.rows;
    const int depth = raw.depth();

    // Get Bayer pattern offsets
    int red_row_off = 0, red_col_off = 0;
    getPatternOffset(pattern, red_row_off, red_col_off);

    // ========================================================================
    // Step 1: Convert to float and create padded buffer
    // ========================================================================
    cv::Mat rawFloat;
    raw.convertTo(rawFloat, CV_32F);

    PaddedBuffer<float, 4> rawPad;
    rawPad.initFromMat(rawFloat);

    // Working buffers (also padded for neighbor access in Pass 2)
    PaddedBuffer<float, 4> green, red, blue;
    green.init(width, height);
    red.init(width, height);
    blue.init(width, height);

    // ========================================================================
    // Step 2: Copy known values to their channels
    // ========================================================================
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rawRow = rawPad.row(y);
        float* gRow = green.row(y);
        float* rRow = red.row(y);
        float* bRow = blue.row(y);

        for (int x = 0; x < width; ++x) {
            float val = rawRow[x];
            PixelType pt = getPixelType(x, y, red_row_off, red_col_off);
            
            switch (pt) {
                case PixelType::Red:
                    rRow[x] = val;
                    break;
                case PixelType::Blue:
                    bRow[x] = val;
                    break;
                case PixelType::GreenOnRed:
                case PixelType::GreenOnBlue:
                    gRow[x] = val;
                    break;
            }
        }
    }

    // ========================================================================
    // Step 3: PASS 1 - Green Interpolation at R/B locations
    // Using directional gradients with branch-free weighted average
    // ========================================================================
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        // Get row pointers for 5-row stencil
        const float* p_m2 = rawPad.row(y - 2);
        const float* p_m1 = rawPad.row(y - 1);
        const float* p    = rawPad.row(y);
        const float* p_p1 = rawPad.row(y + 1);
        const float* p_p2 = rawPad.row(y + 2);
        float* gRow = green.row(y);

        int x = 0;

#if USE_AVX2
        // Process 8 pixels at a time with AVX2
        // Note: We process all pixels and will overwrite Green pixels later
        // This is faster than branching for each pixel
        for (; x + 7 < width; x += 8) {
            __m256 gradH = computeGradH_AVX2(p + x);
            __m256 gradV = computeGradV_AVX2(p_m2 + x, p_m1 + x, p + x, p_p1 + x, p_p2 + x);
            
            // Horizontal interpolation: (left + right)/2 + (2*center - left2 - right2)/4
            __m256 left1 = _mm256_loadu_ps(p + x - 1);
            __m256 right1 = _mm256_loadu_ps(p + x + 1);
            __m256 left2 = _mm256_loadu_ps(p + x - 2);
            __m256 right2 = _mm256_loadu_ps(p + x + 2);
            __m256 center = _mm256_loadu_ps(p + x);
            
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 quarter = _mm256_set1_ps(0.25f);
            __m256 two = _mm256_set1_ps(2.0f);
            
            __m256 avgH = _mm256_mul_ps(_mm256_add_ps(left1, right1), half);
            __m256 corrH = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_mul_ps(two, center), _mm256_add_ps(left2, right2)), 
                quarter
            );
            __m256 estH = _mm256_add_ps(avgH, corrH);
            
            // Vertical interpolation
            __m256 up1 = _mm256_loadu_ps(p_m1 + x);
            __m256 down1 = _mm256_loadu_ps(p_p1 + x);
            __m256 up2 = _mm256_loadu_ps(p_m2 + x);
            __m256 down2 = _mm256_loadu_ps(p_p2 + x);
            
            __m256 avgV = _mm256_mul_ps(_mm256_add_ps(up1, down1), half);
            __m256 corrV = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_mul_ps(two, center), _mm256_add_ps(up2, down2)), 
                quarter
            );
            __m256 estV = _mm256_add_ps(avgV, corrV);
            
            // Branch-free: weighted average based on gradient
            // weight_h = 1 / (grad_h^2 + eps), weight_v = 1 / (grad_v^2 + eps)
            // result = (weight_h * estH + weight_v * estV) / (weight_h + weight_v)
            __m256 eps = _mm256_set1_ps(1e-4f);
            __m256 gradH2 = _mm256_mul_ps(gradH, gradH);
            __m256 gradV2 = _mm256_mul_ps(gradV, gradV);
            __m256 wH = _mm256_rcp_ps(_mm256_add_ps(gradH2, eps));
            __m256 wV = _mm256_rcp_ps(_mm256_add_ps(gradV2, eps));
            
            __m256 sumW = _mm256_add_ps(wH, wV);
            __m256 result = _mm256_div_ps(
                _mm256_add_ps(_mm256_mul_ps(wH, estH), _mm256_mul_ps(wV, estV)),
                sumW
            );
            
            // Store result
            _mm256_storeu_ps(gRow + x, result);
        }
#endif

        // Scalar fallback for remaining pixels
        for (; x < width; ++x) {
            // Skip actual Green pixels
            if (isGreen(x, y, red_row_off, red_col_off)) {
                continue;
            }

            float gradH = computeGradH_Scalar(p, x);
            float gradV = computeGradV_Scalar(p_m2, p_m1, p, p_p1, p_p2, x);
            
            // Horizontal estimate
            float estH = 0.5f * (p[x - 1] + p[x + 1]) + 
                         0.25f * (2.0f * p[x] - p[x - 2] - p[x + 2]);
            
            // Vertical estimate
            float estV = 0.5f * (p_m1[x] + p_p1[x]) + 
                         0.25f * (2.0f * p[x] - p_m2[x] - p_p2[x]);
            
            // Branch-free weighted average
            const float eps = 1e-4f;
            float wH = 1.0f / (gradH * gradH + eps);
            float wV = 1.0f / (gradV * gradV + eps);
            gRow[x] = (wH * estH + wV * estV) / (wH + wV);
        }
    }

    // Restore actual Green values (AVX2 path overwrites them)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rawRow = rawPad.row(y);
        float* gRow = green.row(y);
        for (int x = 0; x < width; ++x) {
            if (isGreen(x, y, red_row_off, red_col_off)) {
                gRow[x] = rawRow[x];
            }
        }
    }

    // Update green buffer padding for Pass 2
    green.updatePadding();

    // ========================================================================
    // Step 4: PASS 2 - Red/Blue Interpolation using Color Differences
    // Key insight: Use color difference (R-G or B-G) which varies more smoothly
    // than raw color values, then reconstruct: R = G + (R-G)
    // ========================================================================
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rawRow = rawPad.row(y);
        const float* rawRowM1 = rawPad.row(y - 1);
        const float* rawRowP1 = rawPad.row(y + 1);
        const float* gRow = green.row(y);
        const float* gRowM1 = green.row(y - 1);
        const float* gRowP1 = green.row(y + 1);
        float* rRow = red.row(y);
        float* bRow = blue.row(y);

        for (int x = 0; x < width; ++x) {
            PixelType pt = getPixelType(x, y, red_row_off, red_col_off);
            float g = gRow[x];

            switch (pt) {
                case PixelType::Red: {
                    // Red is known
                    rRow[x] = rawRow[x];
                    
                    // Interpolate Blue at Red location (diagonal neighbors are Blue)
                    // Use gradient-weighted color difference from diagonals
                    float g_m1_m1 = green.at(x - 1, y - 1);
                    float g_p1_m1 = green.at(x + 1, y - 1);
                    float g_m1_p1 = green.at(x - 1, y + 1);
                    float g_p1_p1 = green.at(x + 1, y + 1);
                    
                    float b_m1_m1 = rawPad.at(x - 1, y - 1);
                    float b_p1_m1 = rawPad.at(x + 1, y - 1);
                    float b_m1_p1 = rawPad.at(x - 1, y + 1);
                    float b_p1_p1 = rawPad.at(x + 1, y + 1);
                    
                    // Compute color differences at diagonal positions
                    float cd_m1_m1 = b_m1_m1 - g_m1_m1;
                    float cd_p1_m1 = b_p1_m1 - g_p1_m1;
                    float cd_m1_p1 = b_m1_p1 - g_m1_p1;
                    float cd_p1_p1 = b_p1_p1 - g_p1_p1;
                    
                    // Gradient-based weighting for diagonal directions
                    float grad_d1 = std::abs(b_m1_m1 - b_p1_p1);  // / diagonal
                    float grad_d2 = std::abs(b_p1_m1 - b_m1_p1);  // \ diagonal
                    
                    const float eps = 1.0f;
                    float w1 = 1.0f / (grad_d1 + eps);
                    float w2 = 1.0f / (grad_d2 + eps);
                    
                    float cd_d1 = (cd_m1_m1 + cd_p1_p1) * 0.5f;
                    float cd_d2 = (cd_p1_m1 + cd_m1_p1) * 0.5f;
                    
                    float avgCD = (w1 * cd_d1 + w2 * cd_d2) / (w1 + w2);
                    bRow[x] = g + avgCD;
                    break;
                }
                
                case PixelType::Blue: {
                    // Blue is known
                    bRow[x] = rawRow[x];
                    
                    // Interpolate Red at Blue location (diagonal neighbors are Red)
                    float g_m1_m1 = green.at(x - 1, y - 1);
                    float g_p1_m1 = green.at(x + 1, y - 1);
                    float g_m1_p1 = green.at(x - 1, y + 1);
                    float g_p1_p1 = green.at(x + 1, y + 1);
                    
                    float r_m1_m1 = rawPad.at(x - 1, y - 1);
                    float r_p1_m1 = rawPad.at(x + 1, y - 1);
                    float r_m1_p1 = rawPad.at(x - 1, y + 1);
                    float r_p1_p1 = rawPad.at(x + 1, y + 1);
                    
                    float cd_m1_m1 = r_m1_m1 - g_m1_m1;
                    float cd_p1_m1 = r_p1_m1 - g_p1_m1;
                    float cd_m1_p1 = r_m1_p1 - g_m1_p1;
                    float cd_p1_p1 = r_p1_p1 - g_p1_p1;
                    
                    float grad_d1 = std::abs(r_m1_m1 - r_p1_p1);
                    float grad_d2 = std::abs(r_p1_m1 - r_m1_p1);
                    
                    const float eps = 1.0f;
                    float w1 = 1.0f / (grad_d1 + eps);
                    float w2 = 1.0f / (grad_d2 + eps);
                    
                    float cd_d1 = (cd_m1_m1 + cd_p1_p1) * 0.5f;
                    float cd_d2 = (cd_p1_m1 + cd_m1_p1) * 0.5f;
                    
                    float avgCD = (w1 * cd_d1 + w2 * cd_d2) / (w1 + w2);
                    rRow[x] = g + avgCD;
                    break;
                }
                
                case PixelType::GreenOnRed: {
                    // Green on Red row: horizontal neighbors are Red, vertical are Blue
                    // Interpolate Red from left/right neighbors
                    float r_m1 = rawRow[x - 1];  // Red at x-1
                    float r_p1 = rawRow[x + 1];  // Red at x+1
                    float g_m1 = gRow[x - 1];
                    float g_p1 = gRow[x + 1];
                    
                    float cd_r = ((r_m1 - g_m1) + (r_p1 - g_p1)) * 0.5f;
                    rRow[x] = g + cd_r;
                    
                    // Interpolate Blue from up/down neighbors
                    float b_m1 = rawRowM1[x];  // Blue at y-1
                    float b_p1 = rawRowP1[x];  // Blue at y+1
                    float g_up = gRowM1[x];
                    float g_dn = gRowP1[x];
                    
                    float cd_b = ((b_m1 - g_up) + (b_p1 - g_dn)) * 0.5f;
                    bRow[x] = g + cd_b;
                    break;
                }
                
                case PixelType::GreenOnBlue: {
                    // Green on Blue row: horizontal neighbors are Blue, vertical are Red
                    // Interpolate Blue from left/right neighbors
                    float b_m1 = rawRow[x - 1];
                    float b_p1 = rawRow[x + 1];
                    float g_m1 = gRow[x - 1];
                    float g_p1 = gRow[x + 1];
                    
                    float cd_b = ((b_m1 - g_m1) + (b_p1 - g_p1)) * 0.5f;
                    bRow[x] = g + cd_b;
                    
                    // Interpolate Red from up/down neighbors
                    float r_m1 = rawRowM1[x];
                    float r_p1 = rawRowP1[x];
                    float g_up = gRowM1[x];
                    float g_dn = gRowP1[x];
                    
                    float cd_r = ((r_m1 - g_up) + (r_p1 - g_dn)) * 0.5f;
                    rRow[x] = g + cd_r;
                    break;
                }
            }
        }
    }

    // ========================================================================
    // Step 5: Output - Combine channels into BGR cv::Mat
    // ========================================================================
    const float maxVal = (depth == CV_16U) ? 65535.0f : 255.0f;
    
    cv::Mat result(height, width, CV_32FC3);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const float* rRow = red.row(y);
        const float* gRow = green.row(y);
        const float* bRow = blue.row(y);
        cv::Vec3f* dstRow = result.ptr<cv::Vec3f>(y);
        
        int x = 0;
        
#if USE_AVX2
        // Process 8 pixels at a time
        for (; x + 7 < width; x += 8) {
            // Load R, G, B
            __m256 r = _mm256_loadu_ps(rRow + x);
            __m256 g = _mm256_loadu_ps(gRow + x);
            __m256 b = _mm256_loadu_ps(bRow + x);
            
            // Clamp to [0, maxVal]
            __m256 zero = _mm256_setzero_ps();
            __m256 maxV = _mm256_set1_ps(maxVal);
            r = _mm256_min_ps(_mm256_max_ps(r, zero), maxV);
            g = _mm256_min_ps(_mm256_max_ps(g, zero), maxV);
            b = _mm256_min_ps(_mm256_max_ps(b, zero), maxV);
            
            // Store as BGR interleaved
            // AVX2 doesn't have great support for interleaving, do scalar
            float rVals[8], gVals[8], bVals[8];
            _mm256_storeu_ps(rVals, r);
            _mm256_storeu_ps(gVals, g);
            _mm256_storeu_ps(bVals, b);
            
            for (int i = 0; i < 8; ++i) {
                dstRow[x + i] = cv::Vec3f(bVals[i], gVals[i], rVals[i]);
            }
        }
#endif
        
        // Scalar fallback
        for (; x < width; ++x) {
            float r = std::clamp(rRow[x], 0.0f, maxVal);
            float g = std::clamp(gRow[x], 0.0f, maxVal);
            float b = std::clamp(bRow[x], 0.0f, maxVal);
            dstRow[x] = cv::Vec3f(b, g, r);  // BGR order for OpenCV
        }
    }
    
    // Convert to output depth
    result.convertTo(dst, CV_MAKETYPE(depth, 3));
}
