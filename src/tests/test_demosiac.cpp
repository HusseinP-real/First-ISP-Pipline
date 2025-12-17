#include "../core/raw_reader.h"
#include "../core/blc.h"
#include "../core/awb.h"
#include "../core/demosiac.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>

int main() {
    // 基础参数
    const std::string inputFile = "data/input/raw1.raw";
    const int width = 512;
    const int height = 500;
    const int frameIndex = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "Demosaic Test - Custom vs OpenCV" << std::endl;
    std::cout << "========================================" << std::endl;

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
    blackLevels bls{135.f, 135.f, 135.f, 135.f};
    applyBlc(reinterpret_cast<uint16_t*>(raw.data), info, bls);
    std::cout << "BLC applied." << std::endl;

    // 3) AWB
    AWBGains gains{1.4f, 1.0f, 1.2f};
    runAWB(raw, gains, false);
    std::cout << "[Manual AWB] Gains: R=" << gains.r << " G=" << gains.g << " B=" << gains.b << std::endl;

    // 4) 使用自定义 demosiac 函数
    std::cout << "\n--- Custom Demosaic (RGGB) ---" << std::endl;
    cv::Mat customColor16;
    demosiac(raw, customColor16, RGGB);
    if (customColor16.empty()) {
        std::cerr << "Custom demosaic output is empty." << std::endl;
        return -1;
    }
    std::cout << "Custom demosaic done. Output type: " << customColor16.type() 
              << " (expected CV_16UC3=" << CV_16UC3 << ")" << std::endl;

    // 统计自定义 demosiac 结果
    double minVal, maxVal;
    cv::minMaxLoc(customColor16, &minVal, &maxVal);
    std::cout << "Custom demosaic value range: [" << (int)minVal << ", " << (int)maxVal << "]" << std::endl;

    // 5) 使用 OpenCV 的 demosiac 作为参考
    std::cout << "\n--- OpenCV Demosaic (RGGB) ---" << std::endl;
    cv::Mat opencvColor16;
    cv::cvtColor(raw, opencvColor16, cv::COLOR_BayerRG2BGR);
    if (opencvColor16.empty()) {
        std::cerr << "OpenCV demosaic output is empty." << std::endl;
        return -1;
    }
    std::cout << "OpenCV demosaic done. Output type: " << opencvColor16.type() << std::endl;

    // 统计 OpenCV demosiac 结果
    double minValCV, maxValCV;
    cv::minMaxLoc(opencvColor16, &minValCV, &maxValCV);
    std::cout << "OpenCV demosaic value range: [" << (int)minValCV << ", " << (int)maxValCV << "]" << std::endl;

    // 6) 转换为 8-bit 用于保存和对比
    const double manualGain = 16.0; // 提升亮度便于查看

    cv::Mat customColor8;
    customColor16.convertTo(customColor8, CV_8UC3, (255.0 / 65535.0) * manualGain);
    
    cv::Mat opencvColor8;
    opencvColor16.convertTo(opencvColor8, CV_8UC3, (255.0 / 65535.0) * manualGain);

    // 7) 计算差异图像
    cv::Mat diff;
    cv::absdiff(customColor8, opencvColor8, diff);
    
    // 将差异图像转换为灰度以便查看
    cv::Mat diffGray;
    cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);
    
    double diffMin, diffMax;
    cv::minMaxLoc(diffGray, &diffMin, &diffMax);
    std::cout << "\nDifference statistics: min=" << (int)diffMin << ", max=" << (int)diffMax << std::endl;

    // 8) 保存结果
    std::string outCustom = "data/output/demosiac_custom_rggb.png";
    std::string outOpenCV = "data/output/demosiac_opencv_rggb.png";
    std::string outDiff = "data/output/demosiac_diff.png";

    if (cv::imwrite(outCustom, customColor8)) {
        std::cout << "Saved custom demosaic: " << outCustom << std::endl;
    } else {
        std::cerr << "Failed to save: " << outCustom << std::endl;
    }

    if (cv::imwrite(outOpenCV, opencvColor8)) {
        std::cout << "Saved OpenCV demosaic: " << outOpenCV << std::endl;
    } else {
        std::cerr << "Failed to save: " << outOpenCV << std::endl;
    }

    // 放大差异图像以便查看（如果差异较小）
    if (diffMax > 0) {
        cv::Mat diffScaled;
        diffGray.convertTo(diffScaled, CV_8U, 255.0 / diffMax);
        if (cv::imwrite(outDiff, diffScaled)) {
            std::cout << "Saved difference map: " << outDiff << std::endl;
        }
    } else {
        std::cout << "No difference detected between custom and OpenCV demosaic." << std::endl;
    }

    // 9) 计算并显示一些统计信息
    cv::Scalar meanCustom = cv::mean(customColor16);
    cv::Scalar meanOpenCV = cv::mean(opencvColor16);
    
    std::cout << "\n--- Statistics Comparison ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Custom demosaic mean:  B=" << meanCustom[0] << ", G=" << meanCustom[1] << ", R=" << meanCustom[2] << std::endl;
    std::cout << "OpenCV demosaic mean:  B=" << meanOpenCV[0] << ", G=" << meanOpenCV[1] << ", R=" << meanOpenCV[2] << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test completed! Check data/output/ folder." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

