 #pragma once
 #include <opencv2/opencv.hpp>
 #include <vector>
 #include <cstdint>
 
 // Gamma 校正类
 // 输入：16-bit RAW 图像（单通道或多通道）
 // 输出：8-bit 图像（通道数与输入一致）
 class GammaCorrection {
 public:
     // 构造时初始化 LUT，默认 gamma=2.2
     explicit GammaCorrection(float gammaVal = 2.2f);
 
     // 运行时更新 gamma，并重建 LUT
     void updateGamma(float gammaVal);
 
     // 对 16-bit RAW 图像做 gamma 校正并输出 8-bit 图像
     // rawImage.depth() 必须为 CV_16U
     void run(const cv::Mat& rawImage, cv::Mat& dst);
 
 private:
     std::vector<uint8_t> lut;  // 0-65535 -> 0-255 的查找表
 
     // 根据 gamma 值生成 LUT
     void createLut16to8(float gammaVal);
 };
