 #pragma once
 #include <opencv2/opencv.hpp>
 #include <vector>
 #include <cstdint>
 
// Gamma 校正类（使用固定的 sRGB 曲线）
// 输入：16-bit RAW 图像（单通道或多通道）
// 输出：8-bit 图像（通道数与输入一致）
class GammaCorrection {
public:
    // 构造时初始化 sRGB LUT
    GammaCorrection();

    // 运行时重建 LUT（可选，因为 sRGB 是固定的）
    void updateGamma();

    // 对 16-bit RAW 图像做 gamma 校正并输出 8-bit 图像
    // rawImage.depth() 必须为 CV_16U
    void run(const cv::Mat& rawImage, cv::Mat& dst);

private:
    std::vector<uint8_t> lut;  // 0-65535 -> 0-255 的查找表

    // 生成固定的 sRGB LUT
    void createLut16to8();
};
