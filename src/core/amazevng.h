#pragma once

#include <opencv2/opencv.hpp>
#include "demosiac.h"

// AMaZE-VNG 混合去马赛克算法
// 结合 VNG 的多方向平均（抗迷宫纹）和 AMaZE 的色差精炼
// 
// 输入:  单通道 Bayer RAW (8/16-bit)
// 输出:  BGR 彩色图 (与输入位深一致)
// 
// 算法流程:
//   1. 使用 VNG 算法获取初始绿色通道（8方向加权，抗迷宫纹）
//   2. 使用 AMaZE 的自适应梯度选择辅助方向决策
//   3. 基于色差的红蓝通道插值
//   4. 中值滤波精炼减少拉链效应
void demosiacAMaZEVNG(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);

