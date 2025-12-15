#include "../core/raw_reader.h"
#include "../core/awb.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

int main() {
    // 1. 读取原始 RAW 图像
    std::string inputFile = "data/input/raw1.raw";
    int frameIndex = 0;
    
    cv::Mat originalRaw = readRawToMat(inputFile, 512, 500, frameIndex);
    
    if (originalRaw.empty()) {
        std::cerr << "Failed to read RAW image: " << inputFile << std::endl;
        return -1;
    }

    std::cout << "Successfully read RAW image. Size: " << originalRaw.cols << "x" << originalRaw.rows << std::endl;
    
    // 检查是否为 16位
    if (originalRaw.type() != CV_16UC1) {
        std::cerr << "Error: Expected 16-bit image." << std::endl;
        return -1;
    }

    // 2. 转换为 8 位（归一化）
    cv::Mat raw8bit;
    originalRaw.convertTo(raw8bit, CV_8U, 255.0 / 65535.0);

    // 3. 进行 Demosaic (Bayer 转 BGR)
    // 假设是 RGGB 模式，如果不是可以改为其他模式
    cv::Mat bgrImage;
    cv::cvtColor(raw8bit, bgrImage, cv::COLOR_BayerRG2BGR);

    std::cout << "Converted to BGR format. Size: " << bgrImage.cols << "x" << bgrImage.rows << std::endl;
    std::cout << "BGR image type: " << bgrImage.type() << " (CV_8UC3 expected)" << std::endl;

    // 4. 定义要测试的手动白平衡增益值
    // 测试两组：r=1.4, g=1, b=1.2 和 r=1.4, g=1, b=1.4
    struct TestGain {
        float r, g, b;
        std::string name;
    };
    
    std::vector<TestGain> testGains = {
        {1.4f, 1.0f, 1.2f, "r1.4_g1_b1.2"},
        {1.4f, 1.0f, 1.4f, "r1.4_g1_b1.4"}
    };

    std::cout << "\n================ START MANUAL AWB TEST ================" << std::endl;

    for (const auto& testGain : testGains) {
        // 每次测试前，从原始 BGR 图像克隆一份
        cv::Mat workImg = bgrImage.clone();

        // 设置手动白平衡增益
        AWBGains gains;
        gains.r = testGain.r;
        gains.g = testGain.g;
        gains.b = testGain.b;

        std::cout << "\n>>> Testing Manual AWB Gains: R=" << testGain.r 
                  << ", G=" << testGain.g << ", B=" << testGain.b << std::endl;

        // 应用手动白平衡 (enableAuto = false)
        runAWB(workImg, gains, false);

        // 统计信息
        cv::Scalar meanScalar = cv::mean(workImg);
        double meanB = meanScalar[0];
        double meanG = meanScalar[1];
        double meanR = meanScalar[2];

        double minVal, maxVal;
        cv::minMaxLoc(workImg, &minVal, &maxVal);

        std::cout << "    Mean Values - B: " << std::fixed << std::setprecision(2) << meanB
                  << ", G: " << meanG << ", R: " << meanR << std::endl;
        std::cout << "    Min: " << (int)minVal << " | Max: " << (int)maxVal << std::endl;

        cv::Mat saveImg;
        workImg.convertTo(saveImg, CV_8UC3, 8.0);

        std::string outputFile = "data/output/awb_manual_" + testGain.name + "_gain4x.png";
        bool success = cv::imwrite(outputFile, saveImg);
        
        if (success) {
            std::cout << "    Saved visual check image to: " << outputFile << " (with 4x gain)" << std::endl;
        } else {
            std::cerr << "    Failed to save image to: " << outputFile << std::endl;
        }
        
        std::cout << "----------------------------------------------" << std::endl;
    }

    std::cout << "\n================ TEST COMPLETED ================" << std::endl;
    return 0;
}

