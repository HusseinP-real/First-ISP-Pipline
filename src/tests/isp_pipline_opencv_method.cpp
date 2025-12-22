#include "../core/raw_reader.h"
#include "../core/blc.h"
#include "../core/denoise.h"
#include "../core/awb.h"
#include "../core/gamma.h"
#include "../core/ccm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

int main() {
    const std::string inputFile = "data/input/raw1.raw";
    const int width = 512;
    const int height = 500;
    const int frameIndex = 0;

    // 1) 读取 RAW (16-bit 单通道，BGGR)
    cv::Mat raw = readRawToMat(inputFile, width, height, frameIndex);
    if (raw.empty()) {
        std::cerr << "Failed to read RAW image: " << inputFile << std::endl;
        return -1;
    }
    if (raw.type() != CV_16UC1) {
        std::cerr << "Error: Expected CV_16UC1 raw input." << std::endl;
        return -1;
    }
    std::cout << "Raw loaded. size=" << raw.cols << "x" << raw.rows << std::endl;

    // 2) 黑电平校正 BLC
    imageInfo info{width, height, 0};
    blackLevels bls{200.f, 200.f, 200.f, 200.f};
    applyBlc(reinterpret_cast<uint16_t*>(raw.data), info, bls);
    std::cout << "BLC applied." << std::endl;

    // 3) Denoise (降噪) - 在RAW域进行，BLC之后、AWB之前
    std::cout << "Applying Denoise..." << std::endl;
    std::vector<uint16_t> raw_vector(width * height);
    std::vector<uint16_t> raw_denoised;
    
    // 将 cv::Mat 转换为 vector<uint16_t>
    const uint16_t* raw_data = raw.ptr<uint16_t>(0);
    std::copy(raw_data, raw_data + width * height, raw_vector.begin());
    
    // 执行降噪
    runDenoise(raw_vector, raw_denoised, width, height);
    
    // 将降噪后的 vector 转换回 cv::Mat
    uint16_t* raw_output_data = raw.ptr<uint16_t>(0);
    std::copy(raw_denoised.begin(), raw_denoised.end(), raw_output_data);
    std::cout << "Denoise applied." << std::endl;

    // 4) AWB
    AWBGains gains{1.4f, 1.0f, 1.2f};
    runAWB(raw, gains, false);
    std::cout << "[Manual AWB] Gains: R=" << gains.r << " G=" << gains.g << " B=" << gains.b << std::endl;

    // 5) Demosaic (使用 OpenCV，BGGR -> BGR，保持 16-bit)
    cv::Mat color16;
    cv::cvtColor(raw, color16, cv::COLOR_BayerBG2BGR);
    std::cout << "Demosaic done (OpenCV)." << std::endl;

    // 6) CCM: 颜色校正矩阵（本次实验先关闭 CCM，直接用 demosaic 输出做后续处理）
    std::cout << "Applying Color Correction Matrix (CCM)..." << std::endl;
    
    float ccm_matrix[3][3] = {
        {1.546875f, -0.29296875f, -0.15625f},      // R_out
        {-0.3125f, 1.71484375f, -0.40625f},        // G_out
        {-0.2265625f, -0.3515625f, 1.609375f}      // B_out
    };

    // 创建 CCM 对象
    ColorCorrectionMatrix ccm(ccm_matrix, 16);  // 16-bit depth

    // 将 OpenCV Mat (BGR) 转换为 vector<uint16_t> (RGB)
    // CCM 期望 RGB 格式，而 demosaic 输出的是 BGR 格式，需要转换
    std::vector<uint16_t> src_rgb;
    std::vector<uint16_t> dst_rgb;
    
    int pixel_count = color16.rows * color16.cols;
    src_rgb.reserve(pixel_count * 3);

    // 转换 BGR -> RGB
    for (int y = 0; y < color16.rows; y++) {
        const uint16_t* row = color16.ptr<uint16_t>(y);
        for (int x = 0; x < color16.cols; x++) {
            // demosaic 输出 BGR: [B, G, R] at indices [0, 1, 2]
            // CCM 期望 RGB: [R, G, B]
            src_rgb.push_back(row[3*x + 2]);  // R (from BGR index 2)
            src_rgb.push_back(row[3*x + 1]);  // G (from BGR index 1)
            src_rgb.push_back(row[3*x + 0]);  // B (from BGR index 0)
        }
    }

    // 应用 CCM
    ccm.process(src_rgb, dst_rgb);
    std::cout << "CCM applied. Processed " << pixel_count << " pixels." << std::endl;

    // 将处理后的 RGB 数据转换回 OpenCV Mat (BGR)
    cv::Mat color16_ccm = cv::Mat::zeros(color16.rows, color16.cols, CV_16UC3);
    for (int y = 0; y < color16_ccm.rows; y++) {
        uint16_t* row = color16_ccm.ptr<uint16_t>(y);
        for (int x = 0; x < color16_ccm.cols; x++) {
            int idx = (y * color16_ccm.cols + x) * 3;
            // CCM 输出 RGB，转换为 OpenCV BGR
            row[3*x + 0] = dst_rgb[idx + 2];  // B (from RGB index 2)
            row[3*x + 1] = dst_rgb[idx + 1];  // G (from RGB index 1)
            row[3*x + 2] = dst_rgb[idx + 0];  // R (from RGB index 0)
        }
    }

    // 7) Digital Gain (在 16-bit 阶段应用，保留更多精度和高光细节)
    // 在 16-bit 阶段应用 gain 可以保留更多动态范围，避免在 8-bit 阶段损失精度
    // 降低数字增益，通过gamma调整来补偿亮度
    // Digital gain（设为 1.0 等价于“去掉增益”）
    double gain = 1.0;
    cv::Mat color16_gain;
    color16_ccm.convertTo(color16_gain, CV_16UC3, gain);
    std::cout << "Digital gain applied (16-bit): " << gain << "x" << std::endl;

    // 8) Demosaic 之后：先把线性 16-bit 量化到 8-bit（不做 gamma）
    //    - 用自适应 scale 避免"把 10/12-bit 当 16-bit"导致整体偏暗
    //    - 用抖动量化减少断层/色带
    const float gamma_value = 2.6f;  // gamma=2.6（增大 gamma 让图像变暗）
    GammaCorrection gamma(gamma_value);
    cv::Mat color8_linear;
    constexpr float kWhiteLevel = 1023.0f; // raw1 很像 10-bit
    const float scale16To8 = 255.0f / (kWhiteLevel * static_cast<float>(gain));
    gamma.quantize16to8WithDithering(color16_gain, color8_linear, scale16To8);

    // 9) Gamma（8-bit -> 8-bit）
    cv::Mat color8_gamma;
    gamma.run8bit(color8_linear, color8_gamma);
    if (color8_gamma.empty()) {
        std::cerr << "Gamma output is empty." << std::endl;
        return -1;
    }
    std::cout << "Gamma applied (8-bit -> 8-bit, gamma=" << gamma_value << ")." << std::endl;

    // 10) 输出 8-bit PNG
    std::string outFile = "data/output/raw1_pipeline_gamma.png";
    if (cv::imwrite(outFile, color8_gamma)) {
        std::cout << "Saved (gamma only, 8-bit): " << outFile << std::endl;
    } else {
        std::cerr << "Failed to save: " << outFile << std::endl;
    }

    return 0;
}
