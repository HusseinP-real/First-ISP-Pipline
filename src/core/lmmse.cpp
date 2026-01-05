/**
 * LMMSE (Linear Minimum Mean Square Error) Demosaicing Algorithm
 * 
 * High-performance implementation for BGGR Bayer pattern
 * Resolution: 500x516, 16-bit unsigned integer input
 * 
 * Author: Expert C++ Image Processing Engineer
 * Standard: C++17
 */

#include "lmmse.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <omp.h>

// BGGR Pattern Layout:
// Row 0: B G B G B G ...
// Row 1: G R G R G R ...
// Row 2: B G B G B G ...
// Row 3: G R G R G R ...

namespace {

enum class BayerColor { RED, GREEN_R, GREEN_B, BLUE };

/**
 * Determine the color at a given pixel position for BGGR pattern
 */
inline BayerColor get_bayer_color(int x, int y) {
    bool even_row = (y % 2 == 0);
    bool even_col = (x % 2 == 0);
    
    if (even_row) {
        // Row 0, 2, 4... : B G B G
        return even_col ? BayerColor::BLUE : BayerColor::GREEN_B;
    } else {
        // Row 1, 3, 5... : G R G R
        return even_col ? BayerColor::GREEN_R : BayerColor::RED;
    }
}

/**
 * Safe pixel access with boundary clamping
 */
inline float safe_access(const float* img, int x, int y, int width, int height) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return img[y * width + x];
}

/**
 * Safe pixel access for input buffer
 */
inline float safe_input(const uint16_t* input, int x, int y, int width, int height) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return static_cast<float>(input[y * width + x]);
}

/**
 * Step 1: Pre-interpolation - Generate initial estimates for R, G, B channels
 */
void pre_interpolate(const uint16_t* input, 
                     float* R, float* G, float* B,
                     int width, int height) {
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            BayerColor color = get_bayer_color(x, y);
            float pixel_val = static_cast<float>(input[idx]);
            
            switch (color) {
                case BayerColor::BLUE: {
                    // At Blue position: B is known, interpolate G and R
                    B[idx] = pixel_val;
                    
                    // Green: average of 4 neighbors (or fewer at edges)
                    float g_sum = 0.0f;
                    int g_count = 0;
                    if (x > 0) { g_sum += safe_input(input, x-1, y, width, height); g_count++; }
                    if (x < width-1) { g_sum += safe_input(input, x+1, y, width, height); g_count++; }
                    if (y > 0) { g_sum += safe_input(input, x, y-1, width, height); g_count++; }
                    if (y < height-1) { g_sum += safe_input(input, x, y+1, width, height); g_count++; }
                    G[idx] = (g_count > 0) ? g_sum / g_count : pixel_val;
                    
                    // Red: bilinear from diagonal neighbors
                    float r_sum = 0.0f;
                    int r_count = 0;
                    if (x > 0 && y > 0) { r_sum += safe_input(input, x-1, y-1, width, height); r_count++; }
                    if (x < width-1 && y > 0) { r_sum += safe_input(input, x+1, y-1, width, height); r_count++; }
                    if (x > 0 && y < height-1) { r_sum += safe_input(input, x-1, y+1, width, height); r_count++; }
                    if (x < width-1 && y < height-1) { r_sum += safe_input(input, x+1, y+1, width, height); r_count++; }
                    R[idx] = (r_count > 0) ? r_sum / r_count : pixel_val;
                    break;
                }
                
                case BayerColor::GREEN_B: {
                    // Green at Blue row: G is known
                    G[idx] = pixel_val;
                    
                    // Blue: horizontal neighbors
                    float b_sum = 0.0f;
                    int b_count = 0;
                    if (x > 0) { b_sum += safe_input(input, x-1, y, width, height); b_count++; }
                    if (x < width-1) { b_sum += safe_input(input, x+1, y, width, height); b_count++; }
                    B[idx] = (b_count > 0) ? b_sum / b_count : pixel_val;
                    
                    // Red: vertical neighbors
                    float r_sum = 0.0f;
                    int r_count = 0;
                    if (y > 0) { r_sum += safe_input(input, x, y-1, width, height); r_count++; }
                    if (y < height-1) { r_sum += safe_input(input, x, y+1, width, height); r_count++; }
                    R[idx] = (r_count > 0) ? r_sum / r_count : pixel_val;
                    break;
                }
                
                case BayerColor::GREEN_R: {
                    // Green at Red row: G is known
                    G[idx] = pixel_val;
                    
                    // Red: horizontal neighbors
                    float r_sum = 0.0f;
                    int r_count = 0;
                    if (x > 0) { r_sum += safe_input(input, x-1, y, width, height); r_count++; }
                    if (x < width-1) { r_sum += safe_input(input, x+1, y, width, height); r_count++; }
                    R[idx] = (r_count > 0) ? r_sum / r_count : pixel_val;
                    
                    // Blue: vertical neighbors
                    float b_sum = 0.0f;
                    int b_count = 0;
                    if (y > 0) { b_sum += safe_input(input, x, y-1, width, height); b_count++; }
                    if (y < height-1) { b_sum += safe_input(input, x, y+1, width, height); b_count++; }
                    B[idx] = (b_count > 0) ? b_sum / b_count : pixel_val;
                    break;
                }
                
                case BayerColor::RED: {
                    // At Red position: R is known, interpolate G and B
                    R[idx] = pixel_val;
                    
                    // Green: average of 4 neighbors
                    float g_sum = 0.0f;
                    int g_count = 0;
                    if (x > 0) { g_sum += safe_input(input, x-1, y, width, height); g_count++; }
                    if (x < width-1) { g_sum += safe_input(input, x+1, y, width, height); g_count++; }
                    if (y > 0) { g_sum += safe_input(input, x, y-1, width, height); g_count++; }
                    if (y < height-1) { g_sum += safe_input(input, x, y+1, width, height); g_count++; }
                    G[idx] = (g_count > 0) ? g_sum / g_count : pixel_val;
                    
                    // Blue: bilinear from diagonal neighbors
                    float b_sum = 0.0f;
                    int b_count = 0;
                    if (x > 0 && y > 0) { b_sum += safe_input(input, x-1, y-1, width, height); b_count++; }
                    if (x < width-1 && y > 0) { b_sum += safe_input(input, x+1, y-1, width, height); b_count++; }
                    if (x > 0 && y < height-1) { b_sum += safe_input(input, x-1, y+1, width, height); b_count++; }
                    if (x < width-1 && y < height-1) { b_sum += safe_input(input, x+1, y+1, width, height); b_count++; }
                    B[idx] = (b_count > 0) ? b_sum / b_count : pixel_val;
                    break;
                }
            }
        }
    }
}

/**
 * Step 2 & 3: Compute difference planes and apply LMMSE filter
 * 
 * LMMSE formula: D_hat = mu + (sigma^2 / (sigma^2 + sigma_noise^2)) * (D_initial - mu)
 * 
 * Uses 5x5 window for local statistics
 */
void apply_lmmse_filter(const float* D_initial, 
                        float* D_filtered,
                        int width, int height,
                        float sigma_noise_sq) {
    
    const int WINDOW_RADIUS = 2;  // 5x5 window
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            // Compute local mean and variance in 5x5 window
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int count = 0;
            
            for (int wy = -WINDOW_RADIUS; wy <= WINDOW_RADIUS; ++wy) {
                for (int wx = -WINDOW_RADIUS; wx <= WINDOW_RADIUS; ++wx) {
                    int nx = std::clamp(x + wx, 0, width - 1);
                    int ny = std::clamp(y + wy, 0, height - 1);
                    float val = D_initial[ny * width + nx];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
            
            // Calculate mean (mu)
            float mu = sum / count;
            
            // Calculate variance (sigma^2) using E[X^2] - E[X]^2
            float variance = (sum_sq / count) - (mu * mu);
            variance = std::max(variance, 0.0f);  // Ensure non-negative
            
            // Apply LMMSE formula: D_hat = mu + (sigma^2 / (sigma^2 + sigma_noise^2)) * (D_initial - mu)
            float D_init_val = D_initial[idx];
            float ratio = variance / (variance + sigma_noise_sq + 1e-10f);  // Small epsilon to avoid division by zero
            
            D_filtered[idx] = mu + ratio * (D_init_val - mu);
        }
    }
}

/**
 * Main LMMSE Demosaicing Function (internal implementation)
 * 
 * @param input     Raw Bayer image (BGGR pattern), single channel
 * @param output    Output RGB image (interleaved R,G,B,R,G,B,...)
 * @param width     Image width
 * @param height    Image height
 * @param sigma_noise  Estimated noise standard deviation
 */
void demosaic_lmmse_impl(const uint16_t* input, 
                        uint16_t* output, 
                        int width, int height, 
                        float sigma_noise) {
    
    const size_t num_pixels = static_cast<size_t>(width) * height;
    const float sigma_noise_sq = sigma_noise * sigma_noise;
    
    // Allocate intermediate buffers for float precision
    std::vector<float> R(num_pixels);
    std::vector<float> G(num_pixels);
    std::vector<float> B(num_pixels);
    std::vector<float> D_R(num_pixels);  // R - G difference
    std::vector<float> D_B(num_pixels);  // B - G difference
    std::vector<float> D_R_filtered(num_pixels);
    std::vector<float> D_B_filtered(num_pixels);
    
    // Step 1: Pre-interpolation - Generate initial R, G, B estimates
    pre_interpolate(input, R.data(), G.data(), B.data(), width, height);
    
    // Step 2: Compute color difference planes
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_pixels; ++i) {
        D_R[i] = R[i] - G[i];
        D_B[i] = B[i] - G[i];
    }
    
    // Step 3 & 4: Apply LMMSE filter to difference planes
    apply_lmmse_filter(D_R.data(), D_R_filtered.data(), width, height, sigma_noise_sq);
    apply_lmmse_filter(D_B.data(), D_B_filtered.data(), width, height, sigma_noise_sq);
    
    // Step 5: Reconstruct final R and B, then output interleaved RGB
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_pixels; ++i) {
        // Final R = G + D_R_filtered
        float final_R = G[i] + D_R_filtered[i];
        // Final B = G + D_B_filtered
        float final_B = G[i] + D_B_filtered[i];
        // G remains as interpolated
        float final_G = G[i];
        
        // Clamp to valid 16-bit range and convert to uint16_t
        final_R = std::clamp(final_R, 0.0f, 65535.0f);
        final_G = std::clamp(final_G, 0.0f, 65535.0f);
        final_B = std::clamp(final_B, 0.0f, 65535.0f);
        
        // Output interleaved RGB
        size_t out_idx = i * 3;
        output[out_idx + 0] = static_cast<uint16_t>(std::round(final_R));
        output[out_idx + 1] = static_cast<uint16_t>(std::round(final_G));
        output[out_idx + 2] = static_cast<uint16_t>(std::round(final_B));
    }
}

} // anonymous namespace

/**
 * Public interface: LMMSE Demosaicing with OpenCV Mat
 */
void demosiacLMMSE(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern, float sigma_noise) {
    // Validate input
    CV_Assert(raw.type() == CV_16UC1);
    CV_Assert(raw.data != nullptr);
    
    // Currently only support BGGR pattern
    if (pattern != BGGR) {
        CV_Error(cv::Error::StsBadArg, "LMMSE demosaicing currently only supports BGGR pattern");
    }
    
    const int width = raw.cols;
    const int height = raw.rows;
    
    // Prepare output image (BGR format, 3 channels, 16-bit)
    dst.create(height, width, CV_16UC3);
    
    // Get input data pointer
    const uint16_t* input_data = reinterpret_cast<const uint16_t*>(raw.data);
    
    // Allocate temporary output buffer (RGB interleaved)
    std::vector<uint16_t> temp_output(static_cast<size_t>(width) * height * 3);
    
    // Call internal implementation
    demosaic_lmmse_impl(input_data, temp_output.data(), width, height, sigma_noise);
    
    // Convert from RGB interleaved to OpenCV BGR format
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t src_idx = (y * width + x) * 3;
            cv::Vec3w& pixel = dst.at<cv::Vec3w>(y, x);
            // OpenCV uses BGR format, so swap R and B
            pixel[0] = temp_output[src_idx + 2];  // B
            pixel[1] = temp_output[src_idx + 1];  // G
            pixel[2] = temp_output[src_idx + 0];  // R
        }
    }
}
