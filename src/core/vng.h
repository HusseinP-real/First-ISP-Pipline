#pragma once
#include <opencv2/opencv.hpp>
#include "demosiac.h"

// VNG (Variable Number of Gradients) demosaic.
// 输入:  Bayer RAW (CV_16UC1 / CV_8UC1)
// 输出:  BGR (CV_16UC3 / CV_8UC3, 与输入位深一致)
//
// 说明:
// - 算法实现按用户提供的 Python 版本逐式翻译，内部默认以 GRBG 为参考排列；
// - 对于其他 Bayer pattern，会先做与 numpy.roll 等价的 2D 循环平移，再在输出端平移回去。
// - 你的传感器如果是 BGGR（左上角为 B），直接传 pattern=BGGR 即可（本项目默认也是 BGGR）。
void demosiacVNG(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern = BGGR);


