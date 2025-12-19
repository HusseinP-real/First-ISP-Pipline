#include "gamma.h"
#include <cmath>



GammaCorrection::GammaCorrection() {
    createLut16to8();
    createLut16to16();
}

void GammaCorrection::updateGamma() {
    createLut16to8();
    createLut16to16();
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
        dstData[i] = lut8[rawData[i]];
    }
}

void GammaCorrection::createLut16to8() {
    // 使用调整后的 gamma 曲线，提供更多亮度补偿
    // 通过使用更小的gamma值（1/2.0而不是1/2.4）来提升整体亮度
    lut8.resize(65536);
    for (int i = 0; i < 65536; ++i) {
        float v = i / 65535.0f; // 归一化到 [0,1]
        float res = 0.0f;

        // 调整后的 gamma 曲线，提供更多亮度
        if (v <= 0.0031308f) {
            // 线性段：抑制暗部噪声，同时保持暗部细节
            res = 12.92f * v;
        } else {
            // 使用更小的gamma值（1/2.0）来提升亮度，补偿减少的数字增益
            res = 1.055f * std::pow(v, 1.0f / 2.0f) - 0.055f;
        }

        lut8[i] = cv::saturate_cast<uint8_t>(res * 255.0f + 0.5f);
    }
}

void GammaCorrection::run16(const cv::Mat& rawImage, cv::Mat& dst) {
    if (rawImage.empty()) return;

    // 只支持 16-bit 输入
    CV_Assert(rawImage.depth() == CV_16U);

    // 输出：16-bit，通道数与输入一致
    dst.create(rawImage.size(), CV_MAKETYPE(CV_16U, rawImage.channels()));

    int totalPixels = rawImage.rows * rawImage.cols;
    int totalValues = rawImage.channels() * totalPixels;

    const uint16_t* rawData = rawImage.ptr<uint16_t>(0);
    uint16_t* dstData = dst.ptr<uint16_t>(0);

    for (int i = 0; i < totalValues; ++i) {
        dstData[i] = lut16[rawData[i]];
    }
}

void GammaCorrection::createLut16to16() {
    // 使用调整后的 gamma 曲线，提供更多亮度补偿
    // 通过使用更小的gamma值（1/2.0而不是1/2.4）来提升整体亮度
    // 输出保持 16-bit 精度 [0, 65535]
    lut16.resize(65536);
    for (int i = 0; i < 65536; ++i) {
        float v = i / 65535.0f; // 归一化到 [0,1]
        float res = 0.0f;

        // 调整后的 gamma 曲线，提供更多亮度
        if (v <= 0.0031308f) {
            res = 12.92f * v;
        } else {
            // 使用更小的gamma值（1/2.0）来提升亮度，补偿减少的数字增益
            res = 1.055f * std::pow(v, 1.0f / 2.0f) - 0.055f;
        }

        // res in [0,1] -> [0,65535]
        int out = static_cast<int>(res * 65535.0f + 0.5f);
        lut16[i] = cv::saturate_cast<uint16_t>(out);
    }
}
