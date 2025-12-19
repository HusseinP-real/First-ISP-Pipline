#include "sharpen.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <stdexcept>

namespace {
// Soft coring: sign(diff) * max(|diff| - threshold, 0)
// Compared to hard thresholding, this avoids abrupt transitions ("banding"/"oil-paint" artifacts).
inline int soft_threshold(int diff, int threshold) {
    if (threshold <= 0) return diff;
    int ad = diff < 0 ? -diff : diff;
    if (ad <= threshold) return 0;
    return diff > 0 ? (ad - threshold) : -(ad - threshold);
}
} // namespace

void sharpen(cv::Mat& img, float amount, int radius, int threshold) {
    if (img.empty() || img.type() != CV_16UC3) {
        throw std::invalid_argument("Image is empty or not 16-bit 3-channel");
    }

    // Parameter hygiene
    if (radius < 0) radius = 0;
    if (threshold < 0) threshold = 0;
    if (amount == 0.0f) return;

    // ISP best-practice: sharpen on luma (Y) to avoid color artifacts / color-noise amplification.
    // Note: cv::cvtColor here is a linear transform; if your pipeline expects sharpening in a specific
    // domain (linear vs gamma), place this module accordingly.
    cv::Mat ycrcb;
    cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);

    // Avoid split/merge (copies 3 channels). We only extract/insert Y (1 channel).
    cv::Mat Y;
    cv::extractChannel(ycrcb, Y, 0);

    // Low frequency (blurred Y)
    cv::Mat blurredY;
    const int ksize = 2 * radius + 1; // 1,3,5...
    // sigma=0 lets OpenCV derive sigma from ksize (stable and easy to tune).
    cv::GaussianBlur(Y, blurredY, cv::Size(ksize, ksize), 0.0);

    // High frequency + coring + add-back (parallel over rows)
    cv::parallel_for_(cv::Range(0, Y.rows), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            uint16_t* yRow = Y.ptr<uint16_t>(y);
            const uint16_t* bRow = blurredY.ptr<uint16_t>(y);
            for (int x = 0; x < Y.cols; ++x) {
                int diff = static_cast<int>(yRow[x]) - static_cast<int>(bRow[x]);
                int cored = soft_threshold(diff, threshold);
                if (cored == 0) continue;
                int newVal = static_cast<int>(yRow[x]) + static_cast<int>(cored * amount);
                yRow[x] = cv::saturate_cast<uint16_t>(newVal);
            }
        }
    });

    cv::insertChannel(Y, ycrcb, 0);
    cv::cvtColor(ycrcb, img, cv::COLOR_YCrCb2BGR);
}

