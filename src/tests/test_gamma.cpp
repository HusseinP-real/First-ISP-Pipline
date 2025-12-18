#include "../core/raw_reader.h"
#include "../core/blc.h"
#include "../core/awb.h"
#include "../core/gamma.h"
#include "../core/demosiac.h"
#include "../core/ccm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // 基础参数
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
    blackLevels bls{135.f, 135.f, 135.f, 135.f}; // 与其他测试保持一致
    applyBlc(reinterpret_cast<uint16_t*>(raw.data), info, bls);
    std::cout << "BLC applied." << std::endl;

    // 3) AWB
    AWBGains gains{1.4f, 1.0f, 1.2f};
    runAWB(raw, gains, false);
    std::cout << "[Manual AWB] Gains: R=" << gains.r << " G=" << gains.g << " B=" << gains.b << std::endl;

    // 4) Demosaic (使用本地实现，BGGR -> BGR，保持 16-bit)
    cv::Mat color16;
    demosiac(raw, color16, BGGR);
    std::cout << "Demosaic done." << std::endl;

    // 5) CCM: 颜色校正矩阵
    std::cout << "Applying Color Correction Matrix (CCM)..." << std::endl;
    
    // 定义 CCM 矩阵
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

    // 6) Digital Gain (在 16-bit 阶段应用，保留更多精度和高光细节)
    // 在 16-bit 阶段应用 gain 可以保留更多动态范围，避免在 8-bit 阶段损失精度
    double gain = 8.0;
    cv::Mat color16_gain;
    // convertTo 会自动将结果 saturate 到 [0, 65535] 范围内
    color16_ccm.convertTo(color16_gain, CV_16UC3, gain);
    std::cout << "Digital gain applied (16-bit): " << gain << "x" << std::endl;

    // 7) Gamma 校正 (16-bit -> 8-bit BGR)
    GammaCorrection gamma;
    cv::Mat color8;
    gamma.run(color16_gain, color8);
    if (color8.empty()) {
        std::cerr << "Gamma output is empty." << std::endl;
        return -1;
    }
    std::cout << "Gamma applied." << std::endl;

    // 8) 保存结果
    std::string outFile = "data/output/raw1_pipeline_gamma.png";
    if (cv::imwrite(outFile, color8)) {
        std::cout << "Saved: " << outFile << std::endl;
    } else {
        std::cerr << "Failed to save: " << outFile << std::endl;
        return -1;
    }

    return 0;
}
