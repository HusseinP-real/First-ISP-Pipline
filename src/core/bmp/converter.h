#ifndef BMP_CONVERTER_H
#define BMP_CONVERTER_H

#include <cstdint>

/**
 * RGB像素结构体
 * 存储RGB三个通道的值，每个通道范围0-255
 */
struct RGBPixel {
    uint8_t R;  // 红色通道 (0-255)
    uint8_t G;  // 绿色通道 (0-255)
    uint8_t B;  // 蓝色通道 (0-255)
};

/**
 * 辅助函数：将值限制在指定范围内
 * 
 * @param value 要限制的值
 * @param min   最小值
 * @param max   最大值
 * @return      限制后的值，在[min, max]范围内
 */
inline uint8_t clamp(int value, int min, int max) {
    if (value < min) {
        return static_cast<uint8_t>(min);
    }
    if (value > max) {
        return static_cast<uint8_t>(max);
    }
    return static_cast<uint8_t>(value);
}

/**
 * RGB到YCbCr颜色空间转换函数
 * 
 * 将RGB图像转换为YCbCr颜色空间，输出三个独立的通道数组
 * 
 * 使用示例：
 * ```cpp
 * #include "converter.h"
 * 
 * // 分配内存
 * RGBPixel** rgb = new RGBPixel*[height];
 * uint8_t** Y = new uint8_t*[height];
 * uint8_t** Cb = new uint8_t*[height];
 * uint8_t** Cr = new uint8_t*[height];
 * 
 * for (int i = 0; i < height; i++) {
 *     rgb[i] = new RGBPixel[width];
 *     Y[i] = new uint8_t[width];
 *     Cb[i] = new uint8_t[width];
 *     Cr[i] = new uint8_t[width];
 * }
 * 
 * // 填充rgb数据...
 * 
 * // 调用转换函数
 * RGB_to_YCbCr(rgb, Y, Cb, Cr, width, height);
 * ```
 * 
 * @param rgb    输入RGB图像，二维数组，每个元素为RGBPixel结构
 * @param Y      输出Y通道（亮度），二维数组，需要预先分配内存
 * @param Cb     输出Cb通道（蓝色色差），二维数组，需要预先分配内存
 * @param Cr     输出Cr通道（红色色差），二维数组，需要预先分配内存
 * @param width  图像宽度
 * @param height 图像高度
 */
void RGB_to_YCbCr(
    RGBPixel** rgb,
    uint8_t** Y,
    uint8_t** Cb,
    uint8_t** Cr,
    int width,
    int height
);

/**
 * YCbCr到RGB颜色空间转换函数
 * 
 * 将YCbCr图像转换回RGB颜色空间
 * 
 * 使用标准ITU-R BT.601逆变换公式：
 * - R = Y + 1.402 * (Cr - 128)
 * - G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 * - B = Y + 1.772 * (Cb - 128)
 * 
 * @param Y      输入Y通道（亮度），二维数组
 * @param Cb     输入Cb通道（蓝色色差），二维数组
 * @param Cr     输入Cr通道（红色色差），二维数组
 * @param rgb    输出RGB图像，二维数组，需要预先分配内存
 * @param width  图像宽度
 * @param height 图像高度
 */
void YCbCr_to_RGB(
    uint8_t** Y,
    uint8_t** Cb,
    uint8_t** Cr,
    RGBPixel** rgb,
    int width,
    int height
);

#endif // BMP_CONVERTER_H
