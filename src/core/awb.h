#pragma once
#include <opencv2/opencv.hpp>

struct AWBGains {
    float r;
    float g;
    float b;
};

void runAWB(cv::Mat& rawImage, AWBGains& gains, bool enableAuto);
