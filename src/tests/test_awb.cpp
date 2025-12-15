#include "../core/raw_reader.h"
#include "../core/blc.h"
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

    // 2. 应用黑电平校正 (BLC) - 使用 blackLevel = 135
    int blackLevel = 135;
    std::cout << "\nApplying Black Level Correction (BLC) with value: " << blackLevel << std::endl;
    
    cv::Mat rawAfterBlc = originalRaw.clone();
    imageInfo info;
    info.width = rawAfterBlc.cols;
    info.height = rawAfterBlc.rows;
    info.ob_rows = 0;
    
    blackLevels bls;
    bls.r = (float)blackLevel;
    bls.gr = (float)blackLevel;
    bls.gb = (float)blackLevel;
    bls.b = (float)blackLevel;
    
    uint16_t* rawData = reinterpret_cast<uint16_t*>(rawAfterBlc.data);
    applyBlc(rawData, info, bls);
    
    std::cout << "BLC completed. Image ready for AWB processing." << std::endl;

    // 3. 定义要测试的手动白平衡增益值
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
        // 每次测试前，从 BLC 处理后的 RAW 图像克隆一份
        cv::Mat workImg = rawAfterBlc.clone();

        // 设置手动白平衡增益
        AWBGains gains;
        gains.r = testGain.r;
        gains.g = testGain.g;
        gains.b = testGain.b;

        std::cout << "\n>>> Testing Manual AWB Gains: R=" << testGain.r 
                  << ", G=" << testGain.g << ", B=" << testGain.b << std::endl;

        // 应用手动白平衡 (enableAuto = false) - 在 16位 RAW 上
        runAWB(workImg, gains, false);

        // 4. 统计 AWB 后的 RAW 数据（用于验证效果）
        // 注意：runAWB 在手动模式下不统计，所以这里需要自己统计来查看结果
        long long sumR = 0, sumG = 0, sumB = 0;
        int countR = 0, countG = 0, countB = 0;

        for (int y = 0; y < workImg.rows; y++) {
            const uint16_t* row = workImg.ptr<uint16_t>(y);
            for (int x = 0; x < workImg.cols; x++) {
                uint16_t value = row[x];
                
                // RGGB pattern
                if (y % 2 == 0) {
                    if (x % 2 == 0) {
                        sumR += value;
                        countR++;
                    } else {
                        sumG += value;
                        countG++;
                    }
                } else {
                    if (x % 2 == 0) {
                        sumG += value;
                        countG++;
                    } else {
                        sumB += value;
                        countB++;
                    }
                }
            }
        }

        double meanR = (countR > 0) ? static_cast<double>(sumR) / countR : 0.0;
        double meanG = (countG > 0) ? static_cast<double>(sumG) / countG : 0.0;
        double meanB = (countB > 0) ? static_cast<double>(sumB) / countB : 0.0;

        double minVal, maxVal;
        cv::minMaxLoc(workImg, &minVal, &maxVal);

        std::cout << "    RAW Statistics (after AWB) - R: " << std::fixed << std::setprecision(2) << meanR
                  << ", G: " << meanG << ", B: " << meanB << std::endl;
        std::cout << "    Min: " << (int)minVal << " | Max: " << (int)maxVal << std::endl;

        // 5. 保存可视化图像（仅用于查看），增加可见度选项
        // 全流程保持 16 位，以下仅为可视化输出
        // 5.1 自动拉伸到满幅的 8 位灰度
        cv::Mat raw8bit_autostretch;
        double scale8 = (maxVal > 0) ? 255.0 / maxVal : 1.0;
        workImg.convertTo(raw8bit_autostretch, CV_8U, scale8);
        std::string outAuto8 = "data/output/awb_manual_" + testGain.name + "_raw_autostretch8.png";
        cv::imwrite(outAuto8, raw8bit_autostretch);
        std::cout << "    Saved visualization (auto-stretch 8-bit) to: " << outAuto8 << std::endl;

        // 5.2 手动增益版 8 位灰度（可调整增益倍数）
        double manualGain = 8.0; // 如仍偏暗，可提高到 16/32
        cv::Mat raw8bit_gain;
        workImg.convertTo(raw8bit_gain, CV_8U, (255.0 / 65535.0) * manualGain);
        std::string outGain8 = "data/output/awb_manual_" + testGain.name + "_raw_gain" + std::to_string((int)manualGain) + "x.png";
        cv::imwrite(outGain8, raw8bit_gain);
        std::cout << "    Saved visualization (manual gain " << (int)manualGain << "x) to: " << outGain8 << std::endl;

        // 5.3 16 位自动拉伸（保留动态范围，用于专业查看）
        cv::Mat raw16_autostretch;
        double scale16 = (maxVal > 0) ? 65535.0 / maxVal : 1.0;
        workImg.convertTo(raw16_autostretch, CV_16U, scale16);
        std::string outAuto16 = "data/output/awb_manual_" + testGain.name + "_raw_autostretch16.png";
        cv::imwrite(outAuto16, raw16_autostretch);
        std::cout << "    Saved visualization (auto-stretch 16-bit) to: " << outAuto16 << std::endl;
        
        std::cout << "----------------------------------------------" << std::endl;
    }

    std::cout << "\n================ TEST COMPLETED ================" << std::endl;
    return 0;
}

