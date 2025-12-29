#pragma once

#include <opencv2/opencv.hpp>
#include "demosiac.h"

// AMaZE (Aliasing Minimization and Zipper Elimination) 去马赛克算法
// 基于自适应方向性插值和色彩差分平滑的高质量 demosaic 算法
// 
// 输入:  单通道 Bayer RAW (8/16-bit)
// 输出:  BGR 彩色图 (与输入位深一致)
// 
// 特点:
//   - 自适应梯度方向选择绿色插值方向
//   - 基于色彩差分的红蓝通道插值
//   - 中值滤波精炼减少拉链效应 (zipper artifacts)
void demosiacAMaZE(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);

