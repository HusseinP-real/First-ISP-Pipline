#include "converter.h"
#include <algorithm>

/**
 * RGB到YCbCr颜色空间转换函数实现
 * 
 * 使用标准ITU-R BT.601公式进行转换：
 * - Y  (亮度):   0.299 * R + 0.587 * G + 0.114 * B
 * - Cb (蓝色差): -0.1687 * R - 0.3313 * G + 0.5 * B + 128
 * - Cr (红色差):  0.5 * R - 0.4187 * G - 0.0813 * B + 128
 * 
 * 转换后的值会被限制在[0, 255]范围内
 */
void RGB_to_YCbCr(
    RGBPixel** rgb,
    uint8_t** Y,
    uint8_t** Cb,
    uint8_t** Cr,
    int width,
    int height
) {
    // 遍历图像的每一行
    for (int y = 0; y < height; y++) {
        // 遍历当前行的每一列
        for (int x = 0; x < width; x++) {
            // 获取当前像素的RGB值
            uint8_t R = rgb[y][x].R;
            uint8_t G = rgb[y][x].G;
            uint8_t B = rgb[y][x].B;
            
            // 将uint8_t转换为int以便进行浮点运算，避免溢出
            int R_int = static_cast<int>(R);
            int G_int = static_cast<int>(G);
            int B_int = static_cast<int>(B);
            
            // 计算Y通道（亮度）
            // Y = 0.299 * R + 0.587 * G + 0.114 * B
            float Y_float = 0.299f * R_int + 0.587f * G_int + 0.114f * B_int;
            int Y_int = static_cast<int>(Y_float + 0.5f);  // 四舍五入
            Y[y][x] = clamp(Y_int, 0, 255);
            
            // 计算Cb通道（蓝色色差）
            // Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
            float Cb_float = -0.1687f * R_int - 0.3313f * G_int + 0.5f * B_int + 128.0f;
            int Cb_int = static_cast<int>(Cb_float + 0.5f);  // 四舍五入
            Cb[y][x] = clamp(Cb_int, 0, 255);
            
            // 计算Cr通道（红色色差）
            // Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
            float Cr_float = 0.5f * R_int - 0.4187f * G_int - 0.0813f * B_int + 128.0f;
            int Cr_int = static_cast<int>(Cr_float + 0.5f);  // 四舍五入
            Cr[y][x] = clamp(Cr_int, 0, 255);
        }
    }
}

/**
 * YCbCr到RGB颜色空间转换函数实现
 * 
 * 使用标准ITU-R BT.601逆变换公式：
 * - R = Y + 1.402 * (Cr - 128)
 * - G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 * - B = Y + 1.772 * (Cb - 128)
 * 
 * 转换后的值会被限制在[0, 255]范围内
 */
void YCbCr_to_RGB(
    uint8_t** Y,
    uint8_t** Cb,
    uint8_t** Cr,
    RGBPixel** rgb,
    int width,
    int height
) {
    // 遍历图像的每一行
    for (int y = 0; y < height; y++) {
        // 遍历当前行的每一列
        for (int x = 0; x < width; x++) {
            // 获取当前像素的YCbCr值
            int Y_int = static_cast<int>(Y[y][x]);
            int Cb_int = static_cast<int>(Cb[y][x]) - 128;  // 去除偏移
            int Cr_int = static_cast<int>(Cr[y][x]) - 128;  // 去除偏移
            
            // 计算R通道
            // R = Y + 1.402 * (Cr - 128)
            float R_float = Y_int + 1.402f * Cr_int;
            int R_result = static_cast<int>(R_float + 0.5f);  // 四舍五入
            rgb[y][x].R = clamp(R_result, 0, 255);
            
            // 计算G通道
            // G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
            float G_float = Y_int - 0.34414f * Cb_int - 0.71414f * Cr_int;
            int G_result = static_cast<int>(G_float + 0.5f);  // 四舍五入
            rgb[y][x].G = clamp(G_result, 0, 255);
            
            // 计算B通道
            // B = Y + 1.772 * (Cb - 128)
            float B_float = Y_int + 1.772f * Cb_int;
            int B_result = static_cast<int>(B_float + 0.5f);  // 四舍五入
            rgb[y][x].B = clamp(B_result, 0, 255);
        }
    }
}
