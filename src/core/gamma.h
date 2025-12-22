 #pragma once
 #include <opencv2/opencv.hpp>
 #include <vector>
 #include <cstdint>
 
// Gamma 校正类（使用可配置的 gamma 曲线）
// 输入：16-bit RAW 图像（单通道或多通道）
// 输出：
//  - run():   8-bit 图像（通道数与输入一致）  —— "曲线 + 8-bit 量化"便于直接保存/显示
//  - run16(): 16-bit 图像（通道数与输入一致） —— 仅做曲线，不降位深，便于后续 ISP 模块（如 Sharpen）
class GammaCorrection {
public:
    // 构造时初始化 gamma LUT（默认 gamma=2.4，sRGB 标准）
    // gamma 值说明：
    //   - 减小 gamma（如 2.4→2.0）→ 图像变亮（曲线更平缓，暗部提升更多）
    //   - 增大 gamma（如 2.4→3.0）→ 图像变暗（曲线更陡，暗部压缩更多）
    GammaCorrection(float gamma = 2.4f);

    // 设置新的 gamma 值并重建 LUT
    void setGamma(float gamma);
    
    // 获取当前 gamma 值
    float getGamma() const { return gamma_value; }

    // 运行时重建 LUT（使用当前 gamma 值）
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

    // 仅做“16-bit 线性 -> 8-bit”量化（带抖动，减少断层），不做 gamma。
    // 适用于你想要的顺序：先量化到 8-bit，再对 8-bit 做 gamma。
    //
    // scale16To8: 把线性 16-bit 映射到 0..255 的比例系数。
    // - 如果你的 RAW 实际是 10/12-bit 装在 16-bit 容器里，建议传入 255.0/whiteLevel
    //   或者用图像统计（max/percentile）自适应得到 scale。
    void quantize16to8WithDithering(const cv::Mat& linear16, cv::Mat& dst8, float scale16To8 = 255.0f / 65535.0f);

private:
    float gamma_value;              // 当前 gamma 值（编码 gamma，默认 2.4）
    std::vector<uint8_t> lut8;     // 0-65535 -> 0-255 的查找表
    std::vector<uint16_t> lut16;   // 0-65535 -> 0-65535 的查找表
    std::vector<uint8_t> lut8to8;  // 0-255 -> 0-255 的查找表（8-bit输入）

    // 生成 gamma LUT（使用当前 gamma_value）
    void createLut16to8();
    void createLut16to16();
    void createLut8to8();
};
