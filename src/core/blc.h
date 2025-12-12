#pragma once
#include <cstdint>

// 图像信息结构体
struct imageInfo {
    int width;      // 图像宽度
    int height;     // 图像高度
    int ob_rows;    // 光学黑区行数
};

// 黑电平结构体
struct blackLevels {
    float r;   // 红色通道黑电平
    float gr;  // 绿色-红色通道黑电平
    float gb;  // 绿色-蓝色通道黑电平
    float b;   // 蓝色通道黑电平
};

// 应用黑电平校正
// 对RAW图像应用黑电平校正，根据Bayer模式选择对应的黑电平值进行减法操作
// @param rawImage: RAW图像数据指针（会被修改）
// @param info: 图像信息（宽度、高度等）
// @param bls: 各通道的黑电平值
void applyBlc(uint16_t* rawImage, const imageInfo& info, const blackLevels& bls);

// 动态统计计算黑电平（已注释，保留接口以备后用）
// blackLevels calcDynamicStatistic(const uint16_t* rawImage, imageInfo& info);
