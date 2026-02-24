#ifndef DCT_DENOISING_H
#define DCT_DENOISING_H

#include <vector>
#include <cmath>

// Perform Sliding Window DCT Denoising (Hard Thresholding)
// src: Input image data (normalized 0-1 or 0-255, assumed single channel)
// dst: Output image data
// width, height: Image dimensions
// sigma: Noise standard deviation
// step: Sliding window step size (default 3)
void denoiseDCT(const std::vector<float>& src, 
                std::vector<float>& dst, 
                int width, 
                int height, 
                float sigma,
                int step = 3);

#endif // DCT_DENOISING_H
