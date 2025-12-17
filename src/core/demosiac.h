#pragma once
#include <opencv2/opencv.hpp>

// Bayer 模式枚举
// 约定如下（与标准 OpenCV / 常见 ISP 定义一致）：
//  RGGB:
//    R G
//    G B   （项目当前 RAW 实际为 BGGR，RGGB 仅保留作兼容）
//  BGGR:
//    B G
//    G R
//  GBRG:
//    G B
//    R G
//  GRBG:
//    G R
//    B G
enum bayerPattern {
    RGGB,
    BGGR,
    GBRG,
    GRBG
};

// 对 RAW 图像进行去马赛克处理
// 将单通道的 Bayer 模式 RAW 图像转换为三通道的 BGR 彩色图像
// @param raw: 输入的 16-bit 单通道 RAW 图像（Bayer 模式）
// @param dst: 输出的 16-bit 三通道 BGR 图像
// @param pattern: Bayer 模式，默认为 BGGR（与当前 RAW 文件一致）
void demosiac(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);

