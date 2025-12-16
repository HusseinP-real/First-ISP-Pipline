#include "gamma.h"
#include <cmath>



GammaCorrection::GammaCorrection(float gammaVal) {
    createLut16to8(gammaVal);
}

void GammaCorrection::updateGamma(float gammaVal) {
    createLut16to8(gammaVal);
}

void GammaCorrection::run(const cv::Mat& rawImage, cv::Mat& dst) {
    if (rawImage.empty()) return;

    // 只支持 16-bit 输入
    CV_Assert(rawImage.depth() == CV_16U);

    // 输出：8-bit，通道数与输入一致
    dst.create(rawImage.size(), CV_MAKETYPE(CV_8U, rawImage.channels()));

    int totalPixels = rawImage.rows * rawImage.cols;
    int totalValues = rawImage.channels() * totalPixels;

    const uint16_t* rawData = rawImage.ptr<uint16_t>(0);
    uint8_t* dstData = dst.ptr<uint8_t>(0);

    for (int i = 0; i < totalValues; ++i) {
        // rawData[i] is 0-65535, convert to 0-255 using lut
        dstData[i] = lut[rawData[i]];
    }
}

void GammaCorrection::createLut16to8(float /*gammaVal*/) {
    // 使用固定的 sRGB 曲线，而不是简单的幂函数 gamma
    lut.resize(65536);
    for (int i = 0; i < 65536; ++i) {
        float v = i / 65535.0f; // 归一化到 [0,1]
        float res = 0.0f;

        // sRGB 标准 OETF
        if (v <= 0.0031308f) {
            // 线性段：抑制暗部噪声，同时保持暗部细节
            res = 12.92f * v;
        } else {
            // 非线性段：接近 1/2.4 次幂的 gamma 曲线
            res = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
        }

        lut[i] = cv::saturate_cast<uint8_t>(res * 255.0f + 0.5f);
    }
}
