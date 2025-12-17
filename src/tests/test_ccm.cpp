#include "../core/raw_reader.h"
#include "../core/blc.h"
#include "../core/awb.h"
#include "../core/demosiac.h"
#include "../core/ccm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // ==================== 基础参数 ====================
    const std::string inputFile = "data/input/raw1.raw";
    const int width = 512;
    const int height = 500;
    const int frameIndex = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "ISP Pipeline Test: Reader -> BLC -> AWB -> Demosaic -> CCM" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==================== 1. Reader: 读取 RAW 图像 ====================
    std::cout << "\n[1/5] Reading RAW image..." << std::endl;
    cv::Mat raw = readRawToMat(inputFile, width, height, frameIndex);
    if (raw.empty()) {
        std::cerr << "Failed to read RAW image: " << inputFile << std::endl;
        return -1;
    }
    if (raw.type() != CV_16UC1) {
        std::cerr << "Error: Expected CV_16UC1 raw input." << std::endl;
        return -1;
    }
    std::cout << "  ✓ Raw loaded. Size: " << raw.cols << "x" << raw.rows << std::endl;

    // ==================== 2. BLC: 黑电平校正 ====================
    std::cout << "\n[2/5] Applying Black Level Correction (BLC)..." << std::endl;
    imageInfo info{width, height, 0};
    blackLevels bls{145.f, 145.f, 145.f, 145.f};
    applyBlc(reinterpret_cast<uint16_t*>(raw.data), info, bls);
    std::cout << "  ✓ BLC applied. Black levels: R=" << bls.r 
              << ", Gr=" << bls.gr << ", Gb=" << bls.gb << ", B=" << bls.b << std::endl;

    // ==================== 3. AWB: 自动白平衡 ====================
    std::cout << "\n[3/5] Applying Auto White Balance (AWB)..." << std::endl;
    AWBGains gains{1.4f, 1.0f, 1.1f};
    runAWB(raw, gains, false);  // false = manual mode, use provided gains
    std::cout << "  ✓ AWB applied. Gains: R=" << gains.r 
              << ", G=" << gains.g << ", B=" << gains.b << std::endl;

    // ==================== 4. Demosaic: 去马赛克 ====================
    std::cout << "\n[4/5] Applying Demosaic..." << std::endl;
    cv::Mat color16;  // 16-bit BGR image (OpenCV 标准格式)
    demosiac(raw, color16, RGGB);
    if (color16.empty()) {
        std::cerr << "  ✗ Demosaic output is empty." << std::endl;
        return -1;
    }
    
    // ========== 检查点 1: 数据类型验证 ==========
    if (color16.type() != CV_16UC3) {
        std::cerr << "  ✗ Fatal Error: Demosaic output type mismatch!" << std::endl;
        std::cerr << "    Expected: " << CV_16UC3 << " (CV_16UC3 = 16-bit 3-channel)" << std::endl;
        std::cerr << "    Actual:   " << color16.type() << " (CV_8UC3=" << CV_8UC3 
                  << ", CV_32FC3=" << CV_32FC3 << ")" << std::endl;
        return -1;
    }
    std::cout << "  ✓ Demosaic done. Output: " << color16.cols << "x" << color16.rows 
              << " (16-bit BGR, type=" << color16.type() << ")" << std::endl;

    // ==================== 5. CCM: 颜色校正矩阵 ====================
    std::cout << "\n[5/5] Applying Color Correction Matrix (CCM)..." << std::endl;
    
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
    
    // ========== 检查点 2: 验证颜色通道顺序 ==========
    // 先保存一张 CCM 处理前的图像，用于验证颜色通道是否正确
    cv::Mat color8_before_ccm_check;
    color16.convertTo(color8_before_ccm_check, CV_8U, 255.0 / 65535.0);
    cv::imwrite("data/output/debug_before_ccm_check.png", color8_before_ccm_check);
    std::cout << "  [Debug] Saved pre-CCM image for channel order verification: debug_before_ccm_check.png" << std::endl;
    
    // 打印前几个像素的 BGR 值用于验证
    std::cout << "  [Debug] First 3 pixels BGR values:" << std::endl;
    for (int y = 0; y < std::min(1, color16.rows); y++) {
        const uint16_t* row = color16.ptr<uint16_t>(y);
        for (int x = 0; x < std::min(3, color16.cols); x++) {
            std::cout << "    Pixel[" << x << "]: B=" << row[3*x + 0] 
                      << ", G=" << row[3*x + 1] << ", R=" << row[3*x + 2] << std::endl;
        }
    }

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
    std::cout << "  Applying CCM..." << std::endl;
    ccm.process(src_rgb, dst_rgb);
    std::cout << "  ✓ CCM applied. Processed " << pixel_count << " pixels." << std::endl;

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
    
    // color16_ccm 已经是 BGR 格式，可以直接使用
    cv::Mat color16_ccm_bgr = color16_ccm;

    // ==================== 保存结果 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Saving results..." << std::endl;

    // 应用数字增益以便更好地查看图像（提升亮度）
    double digital_gain = 64.0;  // 增加增益值使图像更亮
    std::cout << "  Applying digital gain: " << digital_gain << "x" << std::endl;

    // 保存 CCM 处理前的图像（用于对比，color16 已经是 BGR 格式）
    cv::Mat color16_bgr_gain;
    color16.convertTo(color16_bgr_gain, -1, digital_gain);  // 应用增益
    cv::Mat color8_before_ccm;
    color16_bgr_gain.convertTo(color8_before_ccm, CV_8U, 255.0 / 65535.0);
    std::string outFile_before = "data/output/raw1_pipeline_before_ccm.png";
    if (cv::imwrite(outFile_before, color8_before_ccm)) {
        std::cout << "  ✓ Saved (before CCM): " << outFile_before << std::endl;
    } else {
        std::cerr << "  ✗ Failed to save: " << outFile_before << std::endl;
    }

    // 保存 CCM 处理后的图像（color16_ccm_bgr 已经是 BGR 格式）
    cv::Mat color16_ccm_bgr_gain;
    color16_ccm_bgr.convertTo(color16_ccm_bgr_gain, -1, digital_gain);  // 应用增益
    cv::Mat color8_after_ccm;
    color16_ccm_bgr_gain.convertTo(color8_after_ccm, CV_8U, 255.0 / 65535.0);
    std::string outFile_after = "data/output/raw1_pipeline_after_ccm.png";
    if (cv::imwrite(outFile_after, color8_after_ccm)) {
        std::cout << "  ✓ Saved (after CCM): " << outFile_after << std::endl;
    } else {
        std::cerr << "  ✗ Failed to save: " << outFile_after << std::endl;
        return -1;
    }

    // ==================== 统计信息 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Statistics:" << std::endl;
    
    // CCM 前的统计（color16 是 BGR 格式）
    double minVal_before, maxVal_before;
    cv::minMaxLoc(color16, &minVal_before, &maxVal_before);
    cv::Scalar mean_before = cv::mean(color16);
    std::cout << "  Before CCM:" << std::endl;
    std::cout << "    Min: " << minVal_before << ", Max: " << maxVal_before << std::endl;
    // color16 是 BGR 格式 [B, G, R]
    std::cout << "    Mean (BGR): [" << mean_before[0] << ", " << mean_before[1] 
              << ", " << mean_before[2] << "]" << std::endl;

    // CCM 后的统计（color16_ccm 是 BGR 格式）
    double minVal_after, maxVal_after;
    cv::minMaxLoc(color16_ccm, &minVal_after, &maxVal_after);
    cv::Scalar mean_after = cv::mean(color16_ccm);
    std::cout << "  After CCM:" << std::endl;
    std::cout << "    Min: " << minVal_after << ", Max: " << maxVal_after << std::endl;
    // color16_ccm 是 BGR 格式 [B, G, R]
    std::cout << "    Mean (BGR): [" << mean_after[0] << ", " << mean_after[1] 
              << ", " << mean_after[2] << "]" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "ISP Pipeline test completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

