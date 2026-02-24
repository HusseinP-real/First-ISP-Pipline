#ifndef NLM_H
#define NLM_H

#include <opencv2/opencv.hpp>


/**
 * @brief Apply Non-Local Means Denoising on a float Luminance channel.
 * 
 * @param y_float Input luminance image (CV_32F), range [0.0, 1.0].
 * @param h_strength Denoising strength (relative to 0-255 scale). Default 3.0.
 * @param templateWindow Size of the template patch. Default 7.
 * @param searchWindow Size of the window used to compute weighted average. Default 21.
 * @return Denoised float luminance image (CV_32F).
 */
cv::Mat denoiseLuminanceNLM(const cv::Mat& y_float, float h_strength = 3.0f, int templateWindow = 7, int searchWindow = 21);

/**
 * @brief Apply Guided Filter Denoising on Chroma channels (Cb/Cr).
 * 
 * Uses the denoised Luminance (Y) channel as the guide to preserve edges
 * while smoothing the chroma channels, preventing color bleeding.
 * 
 * @param guide_Y The structural reference (Denoised Luminance, CV_32F, range [0.0, 1.0]).
 * @param src_chroma The noisy input chroma channel (Cb or Cr, CV_32F, range [0.0, 1.0]).
 * @param radius The window radius. Default 8.
 * @param eps The regularization parameter (will be squared internally). Default 0.02.
 * @return Denoised chroma image (CV_32F).
 */
cv::Mat denoiseChromaGuided(const cv::Mat& guide_Y, const cv::Mat& src_chroma, int radius = 8, float eps = 0.02f);

/**
 * @brief LUT-optimized Bilateral Filter for luminance denoising.
 * 
 * Uses Look-Up Tables for Gaussian weights to accelerate computation.
 * Preserves edges while smoothing flat regions.
 * 
 * @param src Input image (CV_32F), range [0.0, 1.0].
 * @param radius Filter window radius (window size = 2*radius+1).
 * @param sigma_spatial Spatial Gaussian standard deviation.
 * @param sigma_range Range/intensity Gaussian standard deviation.
 * @return Filtered image (CV_32F).
 */
cv::Mat myBilateralFilter(const cv::Mat& src, int radius, float sigma_spatial, float sigma_range);

#endif // NLM_H
