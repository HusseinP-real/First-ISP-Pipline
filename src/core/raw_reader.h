#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// 读取RAW图像文件（支持多帧，默认读取第一帧）
cv::Mat readRawToMat(const std::string& filename, int width, int height, int frameIndex = 0);