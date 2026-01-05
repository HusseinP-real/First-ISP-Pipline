#pragma once

#include <opencv2/opencv.hpp>
#include "demosiac.h"

/**
 * @brief LMMSE (Linear Minimum Mean Square Error) Demosaicing Algorithm
 * 
 * High-performance implementation for BGGR Bayer pattern
 * Resolution: 500x516, 16-bit unsigned integer input
 * 
 * Algorithm steps:
 * 1. Pre-interpolation - Generate initial estimates for R, G, B channels
 * 2. Compute color difference planes (R-G, B-G)
 * 3. Apply LMMSE filter to difference planes
 * 4. Reconstruct final R and B channels
 * 
 * @param raw Input 16-bit single-channel Bayer RAW image
 * @param dst Output 16-bit 3-channel BGR image
 * @param pattern Bayer pattern (currently supports BGGR)
 * @param sigma_noise Estimated noise standard deviation (default: 100.0)
 */
void demosiacLMMSE(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR, float sigma_noise = 100.0f);
