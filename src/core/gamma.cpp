#include "gamma.h"
#include <cmath>
#include <algorithm>



GammaCorrection::GammaCorrection() {
    createLut16to8();
    createLut16to16();
    createLut8to8();
}

void GammaCorrection::updateGamma() {
    createLut16to8();
    createLut16to16();
    createLut8to8();
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
    // 使用标准的 sRGB gamma 曲线 (gamma = 2.4)
    lut8.resize(65536);
    for (int i = 0; i < 65536; ++i) {
        float v = i / 65535.0f; // 归一化到 [0,1]
        float res = 0.0f;

        // sRGB gamma 曲线
        if (v <= 0.0031308f) {
            // 线性段：抑制暗部噪声，同时保持暗部细节
            res = 12.92f * v;
        } else {
            // 使用标准的 gamma 值（1/2.4）
            res = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
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
    // 使用标准的 sRGB gamma 曲线 (gamma = 2.4)
    // 输出保持 16-bit 精度 [0, 65535]
    lut16.resize(65536);
    for (int i = 0; i < 65536; ++i) {
        float v = i / 65535.0f; // 归一化到 [0,1]
        float res = 0.0f;

        // sRGB gamma 曲线
        if (v <= 0.0031308f) {
            res = 12.92f * v;
        } else {
            // 使用标准的 gamma 值（1/2.4）
            res = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
        }

        // res in [0,1] -> [0,65535]
        int out = static_cast<int>(res * 65535.0f + 0.5f);
        lut16[i] = cv::saturate_cast<uint16_t>(out);
    }
}

void GammaCorrection::run8bit(const cv::Mat& rawImage, cv::Mat& dst) {
    if (rawImage.empty()) return;

    // 只支持 8-bit 输入
    CV_Assert(rawImage.depth() == CV_8U);

    // 输出：8-bit，通道数与输入一致
    dst.create(rawImage.size(), CV_MAKETYPE(CV_8U, rawImage.channels()));

    int totalPixels = rawImage.rows * rawImage.cols;
    int totalValues = rawImage.channels() * totalPixels;

    const uint8_t* rawData = rawImage.ptr<uint8_t>(0);
    uint8_t* dstData = dst.ptr<uint8_t>(0);

    for (int i = 0; i < totalValues; ++i) {
        // rawData[i] is 0-255, apply gamma curve using lut8to8
        dstData[i] = lut8to8[rawData[i]];
    }
}

void GammaCorrection::createLut8to8() {
    // 对 8-bit 输入应用 gamma 曲线，输出 8-bit
    // 使用标准的 sRGB gamma 曲线 (gamma = 2.4)
    lut8to8.resize(256);
    for (int i = 0; i < 256; ++i) {
        float v = i / 255.0f; // 归一化到 [0,1]
        float res = 0.0f;

        // sRGB gamma 曲线
        if (v <= 0.0031308f) {
            // 线性段：抑制暗部噪声，同时保持暗部细节
            res = 12.92f * v;
        } else {
            // 使用标准的 gamma 值（1/2.4）
            res = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
        }

        lut8to8[i] = cv::saturate_cast<uint8_t>(res * 255.0f + 0.5f);
    }
}

void GammaCorrection::runWithDithering(const cv::Mat& rawImage, cv::Mat& dst) {
    if (rawImage.empty()) return;

    // 只支持 16-bit 输入
    CV_Assert(rawImage.depth() == CV_16U);
    CV_Assert(rawImage.channels() == 3); // 抖动目前只支持 3 通道

    // 输出：8-bit BGR
    dst.create(rawImage.size(), CV_8UC3);

    // 误差缓冲（float），用于把高位深 gamma 结果量化进 8-bit 容器时消除台阶
    cv::Mat error = cv::Mat::zeros(rawImage.size(), CV_32FC3);

    // 先对 16-bit 线性值应用 sRGB 曲线（使用 lut16 保留精度），
    // 再在映射到 0..255 并量化时做 Floyd-Steinberg 误差扩散
    constexpr float kScale16To8 = 255.0f / 65535.0f;

    for (int y = 0; y < rawImage.rows; y++) {
        const uint16_t* src_row = rawImage.ptr<uint16_t>(y);
        uint8_t* dst_row = dst.ptr<uint8_t>(y);
        float* err_row = error.ptr<float>(y);
        float* next_err_row = (y + 1 < rawImage.rows) ? error.ptr<float>(y + 1) : nullptr;

        for (int x = 0; x < rawImage.cols; x++) {
            for (int c = 0; c < 3; c++) {
                const uint16_t v_linear16 = src_row[x * 3 + c];

                // 1) gamma(16-bit precision): linear16 -> gamma16
                const uint16_t v_gamma16 = lut16[v_linear16];

                // 2) map to [0..255] (float) + accumulated error
                const float target_val = static_cast<float>(v_gamma16) * kScale16To8 + err_row[x * 3 + c];

                // 3) quantize to 8-bit
                int quantized = static_cast<int>(target_val + 0.5f);
                quantized = std::max(0, std::min(255, quantized));
                dst_row[x * 3 + c] = static_cast<uint8_t>(quantized);

                // 4) error diffusion (Floyd-Steinberg)
                const float quant_error = target_val - static_cast<float>(quantized);

                // right
                if (x + 1 < rawImage.cols) {
                    err_row[(x + 1) * 3 + c] += quant_error * (7.0f / 16.0f);
                }

                // down row
                if (next_err_row) {
                    // down-left
                    if (x > 0) {
                        next_err_row[(x - 1) * 3 + c] += quant_error * (3.0f / 16.0f);
                    }
                    // down
                    next_err_row[x * 3 + c] += quant_error * (5.0f / 16.0f);
                    // down-right
                    if (x + 1 < rawImage.cols) {
                        next_err_row[(x + 1) * 3 + c] += quant_error * (1.0f / 16.0f);
                    }
                }
            }
        }
    }
}
