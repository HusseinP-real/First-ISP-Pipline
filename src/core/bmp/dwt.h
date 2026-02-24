#ifndef DWT_DENOISING_H
#define DWT_DENOISING_H

#include <vector>
#include <cmath>

// Perform Multi-level DWT Denoising (BayesShrink)
// src: Input image data
// dst: Output image data
// width, height: Image dimensions (must be power of 2 or handled internally, here we assume padding if needed)
// levels: Number of decomposition levels (e.g. 2 or 3)
void denoiseDWT(const std::vector<float>& src, 
                std::vector<float>& dst, 
                int width, 
                int height, 
                int levels = 2);

#endif // DWT_DENOISING_H
