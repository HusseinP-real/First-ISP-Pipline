#include "denoise.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>

// range weight lut
std::vector<double> init_range_weight_lut(double sigma_r) {
    std::vector<double> lut(65536);
    double coeff = -1.0 / (2.0 * sigma_r * sigma_r);  // Fixed: added negative sign
    for (int i = 0; i < 65536; i++) {
        lut[i] = std::exp(i * i * coeff);
    }
    return lut;
}

// spatial weight lut
std::vector<std::vector<double>> init_spatial_weight_kernel(int radius, double sigma_s) {
    int size = 2 * radius + 1;
    std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
    double coeff = -1.0 / (2.0 * sigma_s * sigma_s);  // Fixed: added negative sign
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            kernel[y + radius][x + radius] = std::exp((x * x + y * y) * coeff);  // Fixed: correct indexing
        }
    }
    return kernel;
}

// single channel denoise
void bilateral_filter_sub_image(const std::vector<uint16_t>& src, std::vector<uint16_t>& dst, 
                        int w, int h, const std::vector<double>& range_lut,
                         const std::vector<std::vector<double>>& spatial_kernel,
                        int radius) {

    // Initialize output with input values (for edge handling)
    dst = src;

    // iterate each pixel (skip edges)
    for (int y = radius; y < h - radius; y++) {
        for (int x = radius; x < w - radius; x++) {
            double sum_weight = 0.0;
            double sum_value = 0.0;
            uint16_t center_val = src[y * w + x];

            // iterate each pixel in the kernel
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    uint16_t neighbor_val = src[(y + ky) * w + (x + kx)];

                    // range weight
                    int diff = std::abs(static_cast<int>(center_val) - static_cast<int>(neighbor_val));
                    if (diff >= static_cast<int>(range_lut.size())) {
                        diff = static_cast<int>(range_lut.size()) - 1;  // Clamp to valid range
                    }
                    double w_r = range_lut[diff];

                    // get spatial weight
                    double w_s = spatial_kernel[ky + radius][kx + radius];
                    
                    double weight = w_r * w_s;
                    
                    sum_value += neighbor_val * weight;
                    sum_weight += weight;
                }
            }
            
            // Avoid division by zero
            if (sum_weight > 0.0) {
                dst[y * w + x] = static_cast<uint16_t>(std::round(sum_value / sum_weight));
            }
        }
    }
    
}


void runDenoise(const std::vector<uint16_t>& raw_input, std::vector<uint16_t>& raw_output, 
                int width, int height) {
    double sigma_s = 2.0;
    double sigma_r = 200.0;
    int radius = 2;

    // Calculate sub-image dimensions (half size for each Bayer channel)
    int SUB_W = width / 2;
    int SUB_H = height / 2;

    // Validate input size
    if (raw_input.size() != static_cast<size_t>(width * height)) {
        std::cerr << "Error: Input size mismatch. Expected " << width * height 
                  << " but got " << raw_input.size() << std::endl;
        raw_output = raw_input;  // Return input unchanged on error
        return;
    }

    // Initialize output
    raw_output.resize(width * height);

    // init range weight and spatial kernel lut
    auto rLUT = init_range_weight_lut(sigma_r);
    auto sKernel = init_spatial_weight_kernel(radius, sigma_s);

    std::vector<std::vector<uint16_t>> channels(4, std::vector<uint16_t>(SUB_W * SUB_H));
    std::vector<std::vector<uint16_t>> channels_denoised(4, std::vector<uint16_t>(SUB_W * SUB_H));

    // === Split (De-interleave) ===
    // BGGR: B=0, Gb=1, Gr=2, R=3
    for (int y = 0; y < SUB_H; ++y) {
        for (int x = 0; x < SUB_W; ++x) {
            int b_idx  = (2 * y) * width + (2 * x);      // B
            int gb_idx = (2 * y) * width + (2 * x + 1);  // Gb
            int gr_idx = (2 * y + 1) * width + (2 * x);  // Gr
            int r_idx  = (2 * y + 1) * width + (2 * x + 1);// R

            channels[0][y * SUB_W + x] = raw_input[b_idx];
            channels[1][y * SUB_W + x] = raw_input[gb_idx];
            channels[2][y * SUB_W + x] = raw_input[gr_idx];
            channels[3][y * SUB_W + x] = raw_input[r_idx];
        }
    }

    // === Execute Filter on each channel ===
    for (int k = 0; k < 4; ++k) {
        bilateral_filter_sub_image(channels[k], channels_denoised[k], 
                                SUB_W, SUB_H, rLUT, sKernel, radius);
    }

    // === Merge (Re-interleave) ===
    for (int y = 0; y < SUB_H; ++y) {
        for (int x = 0; x < SUB_W; ++x) {
            int sub_idx = y * SUB_W + x;
            
            raw_output[(2 * y) * width + (2 * x)]       = channels_denoised[0][sub_idx]; // B
            raw_output[(2 * y) * width + (2 * x + 1)]   = channels_denoised[1][sub_idx]; // Gb
            raw_output[(2 * y + 1) * width + (2 * x)]   = channels_denoised[2][sub_idx]; // Gr
            raw_output[(2 * y + 1) * width + (2 * x + 1)] = channels_denoised[3][sub_idx]; // R
        }
    }
}