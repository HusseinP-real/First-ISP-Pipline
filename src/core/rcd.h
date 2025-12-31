#pragma once

#include <opencv2/opencv.hpp>
#include "demosiac.h"

/**
 * @brief RCD (Ratio-based Color Difference) Demosaic Algorithm
 * 
 * High-quality demosaicing algorithm with two core phases:
 * 1. Green interpolation using directional gradients (Hamilton-Adams style)
 * 2. Red/Blue interpolation using color difference ratios
 * 
 * Optimizations:
 * - Pre-padded buffer to eliminate boundary checks
 * - OpenMP parallelization for row processing
 * - AVX2/SSE SIMD for gradient computation
 * - Branch-free gradient direction selection
 * 
 * @param raw Input 16-bit single-channel Bayer RAW image
 * @param dst Output 16-bit 3-channel BGR image
 * @param pattern Bayer pattern (RGGB, BGGR, GBRG, GRBG)
 */
void demosiacRCD(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);

