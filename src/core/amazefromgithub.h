#pragma once

#include <opencv2/opencv.hpp>
#include "demosiac.h"

// AMaZE (Aliasing Minimization and Zipper Elimination) 去马赛克算法
// 基于 GitHub 原始实现的完整版本
// (Aliasing Minimization and Zipper Elimination)
//
// 输入:  单通道 Bayer RAW (8/16-bit)
// 输出:  BGR 彩色图 (与输入位深一致)
// 
// 特点:
//   - 分块处理 (Tile-based processing)
//   - 自适应方向性插值
//   - Nyquist 纹理检测和处理
//   - 对角插值修正
//   - 色度精炼
void demosiacAMaZEFromGitHub(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);

