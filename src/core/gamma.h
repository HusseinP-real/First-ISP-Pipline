 #pragma once
 #include <opencv2/opencv.hpp>
 #include <vector>
 #include <cstdint>
 
// Gamma 校正类（使用固定的 sRGB 曲线）
// 输入：16-bit RAW 图像（单通道或多通道）
// 输出：
//  - run():   8-bit 图像（通道数与输入一致）  —— “曲线 + 8-bit 量化”便于直接保存/显示
//  - run16(): 16-bit 图像（通道数与输入一致） —— 仅做曲线，不降位深，便于后续 ISP 模块（如 Sharpen）
class GammaCorrection {
public:
    // 构造时初始化 sRGB LUT
    GammaCorrection();

    // 运行时重建 LUT（可选，因为 sRGB 是固定的）
    void updateGamma();

    // 对 16-bit RAW 图像做 gamma 校正并输出 8-bit 图像
    // rawImage.depth() 必须为 CV_16U
    void run(const cv::Mat& rawImage, cv::Mat& dst);

    // 对 16-bit 图像做 gamma 校正并输出 16-bit 图像（不降位深）
    // rawImage.depth() 必须为 CV_16U
    void run16(const cv::Mat& rawImage, cv::Mat& dst);

    // 对 8-bit 图像做 gamma 校正并输出 8-bit 图像
    // rawImage.depth() 必须为 CV_8U
    void run8bit(const cv::Mat& rawImage, cv::Mat& dst);

    // 16-bit 转 8-bit 并应用 gamma 校正（带抖动，减少色彩断层）
    // rawImage.depth() 必须为 CV_16U
    // 正确顺序：先在高位深(16-bit/float)上应用 sRGB 曲线，再在量化到 8-bit 时做 Floyd-Steinberg 误差扩散
    void runWithDithering(const cv::Mat& rawImage, cv::Mat& dst);

private:
    std::vector<uint8_t> lut8;     // 0-65535 -> 0-255 的查找表
    std::vector<uint16_t> lut16;   // 0-65535 -> 0-65535 的查找表
    std::vector<uint8_t> lut8to8;  // 0-255 -> 0-255 的查找表（8-bit输入）

    // 生成固定的 sRGB LUT
    void createLut16to8();
    void createLut16to16();
    void createLut8to8();
};
