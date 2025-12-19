#pragma once
#include <vector>
#include <cstdint>

// 初始化范围权重查找表（用于双边滤波）
// @param sigma_r: 范围标准差参数
// @return: 预计算的范围权重查找表
std::vector<double> init_range_weight_lut(double sigma_r);

// 初始化空间权重核（用于双边滤波）
// @param radius: 滤波半径
// @param sigma_s: 空间标准差参数
// @return: 预计算的空间权重核
std::vector<std::vector<double>> init_spatial_weight_kernel(int radius, double sigma_s);

// 引导双边滤波（Joint / Guided Bilateral Filter）
// @param src: 需要被平滑/降噪的通道
// @param guide: 引导图（决定“哪里是边缘”），通常为 G 通道
// @param dst: 输出图像数据（一维向量，会被修改）
// @param w: 图像宽度
// @param h: 图像高度
// @param range_lut: 范围权重查找表
// @param spatial_kernel: 空间权重核
// @param radius: 滤波半径
void bilateral_filter_guided(const std::vector<uint16_t>& src,
                             const std::vector<uint16_t>& guide,
                             std::vector<uint16_t>& dst,
                             int w, int h,
                             const std::vector<double>& range_lut,
                             const std::vector<std::vector<double>>& spatial_kernel,
                             int radius);

// 对RAW图像进行降噪处理（BGGR格式）
// 将RAW图像按Bayer模式分离为4个通道，分别进行双边滤波，然后重新合并
// @param raw_input: 输入的RAW图像数据（一维向量，BGGR格式）
// @param raw_output: 输出的降噪后RAW图像数据（一维向量，会被修改）
// @param width: 图像宽度
// @param height: 图像高度
void runDenoise(const std::vector<uint16_t>& raw_input, std::vector<uint16_t>& raw_output,
                int width, int height);

