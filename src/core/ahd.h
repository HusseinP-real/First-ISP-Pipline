#pragma once

#include <opencv2/opencv.hpp>
#include "demosiac.h"

// AHD (Adaptive Homogeneity Directed) 去马赛克
// 输入:  单通道 Bayer RAW (8/16-bit)
// 输出:  BGR 彩色图 (与输入位深一致)
// 说明:  当前实现针对 BGGR 模式做了优化和验证
void demosiacAHD(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);
