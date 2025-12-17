#pragma once
#include <opencv2/opencv.hpp>

// Bayer 模式枚举
enum bayerPattern {
    RGGB,  // Red-Green-Green-Blue (最常用)
    GBBR,  // Green-Blue-Blue-Red
    GRBG,  // Green-Red-Blue-Green
    BGRG   // Blue-Green-Red-Green
};

// 对 RAW 图像进行去马赛克处理
// 将单通道的 Bayer 模式 RAW 图像转换为三通道的 BGR 彩色图像
// @param raw: 输入的 16-bit 单通道 RAW 图像（Bayer 模式）
// @param dst: 输出的 16-bit 三通道 BGR 图像
// @param pattern: Bayer 模式，默认为 RGGB（目前仅支持 RGGB）
void demosiac(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = RGGB);

