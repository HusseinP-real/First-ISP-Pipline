#include "../core/raw_reader.h"
#include "../core/blc.h"
#include "../core/awb.h"
#include "../core/gamma.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    // 基础参数
    const std::string inputFile = "data/input/raw1.raw";
    const int width = 512;
    const int height = 500;
    const int frameIndex = 0;

    // 1) 读取 RAW (16-bit 单通道，RGGB)
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

    // 4) Demosaic (RGGB -> BGR, 保持 16-bit)
    cv::Mat color16;
    cv::cvtColor(raw, color16, cv::COLOR_BayerRG2BGR);
    std::cout << "Demosaic done." << std::endl;

    // 5) Gamma 校正 (16-bit -> 8-bit BGR)
    GammaCorrection gamma(2.2f);
    cv::Mat color8;
    gamma.run(color16, color8);
    if (color8.empty()) {
        std::cerr << "Gamma output is empty." << std::endl;
        return -1;
    }
    std::cout << "Gamma applied." << std::endl;

    // 6)
    cv::Mat color8_gain;
    double gain = 4;
    color8.convertTo(color8_gain, -1, gain);
    std::cout << "Extra digital gain applied: " << gain << "x" << std::endl;

    // 7) 保存结果
    std::string outFile = "data/output/raw1_pipeline_gamma.png";
    if (cv::imwrite(outFile, color8_gain)) {
        std::cout << "Saved: " << outFile << std::endl;
    } else {
        std::cerr << "Failed to save: " << outFile << std::endl;
        return -1;
    }

    return 0;
}
