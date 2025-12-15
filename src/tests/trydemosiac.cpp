#include "../core/awb.h"
#include "../core/raw_reader.h"  
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::string inputFile = "data/input/raw1.raw";
    int width = 512;
    int height = 500;
    int blackLevel = 135; // 已调好的黑电平

    // 1. 读取
    cv::Mat raw = readRawToMat(inputFile, width, height, /*frameIndex=*/0);
    if (raw.empty()) {
        std::cerr << "Error reading file." << std::endl;
        return -1;
    }

    // 2. BLC
    cv::subtract(raw, cv::Scalar(blackLevel), raw);
    std::cout << "Raw loaded and BLC applied." << std::endl;

    // 只测试 RGGB（不再遍历所有 Bayer 模式）
    struct Pattern { int code; std::string name; };
    std::vector<Pattern> patterns = {
        {cv::COLOR_BayerRG2BGR, "RGGB"}
    };

    // 手动 AWB 增益，与 test_awb.cpp 保持一致
    struct ManualGain {
        AWBGains gains;
        std::string name;
    };
    std::vector<ManualGain> manualGains = {
        {{1.4f, 1.0f, 1.2f}, "r1.4_g1_b1.2"},
        {{1.4f, 1.0f, 1.4f}, "r1.4_g1_b1.4"}
    };

    std::cout << "========================================" << std::endl;
    std::cout << "Starting RAW-domain AWB then Demosaic..." << std::endl;

    const double manualGain = 16.0; // 提升亮度便于查看（无 gamma）

    for (const auto& mg : manualGains) {
        std::cout << "\n=== Manual AWB Set: " << mg.name << " ===" << std::endl;

        // 在 RAW 域先做手动 AWB
        cv::Mat awbRaw = raw.clone();
        AWBGains gains = mg.gains;
        runAWB(awbRaw, gains, false);

        for (const auto& p : patterns) {
            std::cout << ">>> Demosaic Pattern: " << p.name << std::endl;

            // 3. Demosaic (16-bit RAW -> 16-bit BGR)
            cv::Mat color16;
            cv::cvtColor(awbRaw, color16, p.code);

            // 4. 转为 8 位便于查看保存（增加手动增益，不做 gamma）
            cv::Mat color8;
            color16.convertTo(color8, CV_8UC3, (255.0 / 65535.0) * manualGain);

            // 5. 保存
            std::string filename = "data/output/ref_opencv_" + mg.name + "_" + p.name + ".png";
            cv::imwrite(filename, color8);
            std::cout << "    Saved: " << filename << std::endl;
        }
    }

    std::cout << "Done! Please check data/output/ folder." << std::endl;
    return 0;
}

