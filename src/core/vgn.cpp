#include "vgn.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// 辅助函数：快速获取绝对值
inline float absf(float a) { return std::fabs(a); }

// 辅助函数：计算 VNG 阈值
inline float get_threshold(const float gradients[8]) {
    float min_g = std::numeric_limits<float>::max();
    float max_g = -std::numeric_limits<float>::max();

    // 编译器通常会自动展开这个小循环
    for (int i = 0; i < 8; ++i) {
        min_g = std::min(min_g, gradients[i]);
        max_g = std::max(max_g, gradients[i]);
    }
    // Python公式: T = min + 0.5*max
    return min_g + 0.5f * max_g;
}

// 仅支持 BGGR 的快速 VNG
static void vng_demosaic_bggr(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_16UC1 || src.type() == CV_8UC1);

    const int rows = src.rows;
    const int cols = src.cols;
    CV_Assert(rows >= 5 && cols >= 5);

    // 1) 预处理：转为 float 以保证精度
    cv::Mat raw_f;
    src.convertTo(raw_f, CV_32F);

    // 2) 初始化输出：先用灰度(=raw)填满整张图，保证边界像素不会是未初始化垃圾值
    //    （对应原 Python：out = inp.copy()，且 inp 在 Bayer 场景下可看作“每个通道先等于 raw”）
    cv::Mat out_f;
    {
        std::vector<cv::Mat> ch(3, raw_f);
        cv::merge(ch, out_f); // CV_32FC3
    }

    // 3) 核心循环：忽略边缘 2 像素 (VNG 需要 5x5 窗口)
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 2; y < rows - 2; ++y) {
        // 获取当前行及邻域行的指针，减少重复的指针运算
        const float* r_m2 = raw_f.ptr<float>(y - 2);
        const float* r_m1 = raw_f.ptr<float>(y - 1);
        const float* r_0  = raw_f.ptr<float>(y);
        const float* r_p1 = raw_f.ptr<float>(y + 1);
        const float* r_p2 = raw_f.ptr<float>(y + 2);

        cv::Vec3f* out_ptr = out_f.ptr<cv::Vec3f>(y);

        for (int x = 2; x < cols - 2; ++x) {
            float val = r_0[x]; // 当前像素值
            
            // BGGR 模式判定:
            // (0,0) -> B, (0,1) -> G(R行), (1,0) -> G(B行), (1,1) -> R
            // y%2==0, x%2==0 : Blue
            // y%2==0, x%2==1 : Green (在 Blue 行) -> 左右是 B, 上下是 R
            // y%2==1, x%2==0 : Green (在 Red 行)  -> 左右是 R, 上下是 B
            // y%2==1, x%2==1 : Red

            bool is_blue  = ((y & 1) == 0) && ((x & 1) == 0);
            bool is_green_b = ((y & 1) == 0) && ((x & 1) == 1); // Green on Blue row
            bool is_green_r = ((y & 1) == 1) && ((x & 1) == 0); // Green on Red row
            bool is_red   = ((y & 1) == 1) && ((x & 1) == 1);

            float gra[8]; // N, E, S, W, NE, SE, NW, SW

            // ================= 梯度计算 =================
            // 注意：VNG 的梯度公式虽然长，但对 R 和 B 是对称的。
            // 只有 Green 处理时，根据它是 "Red行Green" 还是 "Blue行Green" 有细微区别。
            
            if (is_red || is_blue) { 
                // ---- Red 或 Blue 像素的处理逻辑 (对称) ----
                // 对于 BGGR:
                // 如果是 Red (y,x)，则 inR=val, inG=?, inB=?
                // 邻域: 上下左右是 G，对角线是 B (Red点) 或 R (Blue点)
                
                // 下面的 index 偏移量是固定的，直接硬编码计算
                // N
                gra[0] = absf(r_m1[x] - r_p1[x]) + absf(r_m2[x] - val) + 
                         0.5f * absf(r_m1[x-1] - r_p1[x-1]) + 
                         0.5f * absf(r_m1[x+1] - r_p1[x+1]) + 
                         0.5f * absf(r_m2[x-1] - r_0[x-1]) + 
                         0.5f * absf(r_m2[x+1] - r_0[x+1]);
                // E
                gra[1] = absf(r_0[x+1] - r_0[x-1]) + absf(r_0[x+2] - val) +
                         0.5f * absf(r_m1[x+1] - r_m1[x-1]) +
                         0.5f * absf(r_p1[x+1] - r_p1[x-1]) +
                         0.5f * absf(r_m1[x+2] - r_m1[x]) +
                         0.5f * absf(r_p1[x+2] - r_p1[x]);
                // S
                gra[2] = absf(r_p1[x] - r_m1[x]) + absf(r_p2[x] - val) +
                         0.5f * absf(r_p1[x+1] - r_m1[x+1]) +
                         0.5f * absf(r_p1[x-1] - r_m1[x-1]) +
                         0.5f * absf(r_p2[x+1] - r_0[x+1]) +
                         0.5f * absf(r_p2[x-1] - r_0[x-1]);
                // W
                gra[3] = absf(r_0[x-1] - r_0[x+1]) + absf(r_0[x-2] - val) +
                         0.5f * absf(r_p1[x-1] - r_p1[x+1]) +
                         0.5f * absf(r_m1[x-1] - r_m1[x+1]) +
                         0.5f * absf(r_p1[x-2] - r_p1[x]) +
                         0.5f * absf(r_m1[x-2] - r_m1[x]);
                
                // NE
                gra[4] = absf(r_m1[x+1] - r_p1[x-1]) + absf(r_m2[x+2] - val) +
                         0.5f * absf(r_m1[x] - r_0[x-1]) + 0.5f * absf(r_0[x+1] - r_p1[x]) +
                         0.5f * absf(r_m2[x+1] - r_m1[x]) + 0.5f * absf(r_m1[x+2] - r_0[x+1]);
                // SE
                gra[5] = absf(r_p1[x+1] - r_m1[x-1]) + absf(r_p2[x+2] - val) +
                         0.5f * absf(r_0[x+1] - r_m1[x]) + 0.5f * absf(r_p1[x] - r_0[x-1]) +
                         0.5f * absf(r_p1[x+2] - r_0[x+1]) + 0.5f * absf(r_p2[x+1] - r_p1[x]);
                // NW
                gra[6] = absf(r_m1[x-1] - r_p1[x+1]) + absf(r_m2[x-2] - val) +
                         0.5f * absf(r_0[x-1] - r_p1[x]) + 0.5f * absf(r_m1[x] - r_0[x+1]) +
                         0.5f * absf(r_m1[x-2] - r_0[x-1]) + 0.5f * absf(r_m2[x-1] - r_m1[x]);
                // SW
                gra[7] = absf(r_p1[x-1] - r_m1[x+1]) + absf(r_p2[x-2] - val) +
                         0.5f * absf(r_p1[x] - r_0[x+1]) + 0.5f * absf(r_0[x-1] - r_m1[x]) +
                         0.5f * absf(r_p2[x-1] - r_p1[x]) + 0.5f * absf(r_p1[x-2] - r_0[x-1]);

            } else { 
                // ---- Green 像素的处理逻辑 ----
                // 梯度计算公式对于两种 Green 其实是通用的，区别在于它计算的是“同色”和“异色”差异
                
                // N
                gra[0] = absf(r_m1[x] - r_p1[x]) + absf(r_m2[x] - val) +
                         0.5f * absf(r_m1[x-1] - r_p1[x-1]) + 0.5f * absf(r_m1[x+1] - r_p1[x+1]) +
                         0.5f * absf(r_m2[x-1] - r_0[x-1]) + 0.5f * absf(r_m2[x+1] - r_0[x+1]);
                // E
                gra[1] = absf(r_0[x+1] - r_0[x-1]) + absf(r_0[x+2] - val) +
                         0.5f * absf(r_m1[x+1] - r_m1[x-1]) + 0.5f * absf(r_p1[x+1] - r_p1[x-1]) +
                         0.5f * absf(r_m1[x+2] - r_m1[x]) + 0.5f * absf(r_p1[x+2] - r_p1[x]);
                // S
                gra[2] = absf(r_p1[x] - r_m1[x]) + absf(r_p2[x] - val) +
                         0.5f * absf(r_p1[x+1] - r_m1[x+1]) + 0.5f * absf(r_p1[x-1] - r_m1[x-1]) +
                         0.5f * absf(r_p2[x+1] - r_0[x+1]) + 0.5f * absf(r_p2[x-1] - r_0[x-1]);
                // W
                gra[3] = absf(r_0[x-1] - r_0[x+1]) + absf(r_0[x-2] - val) +
                         0.5f * absf(r_p1[x-1] - r_p1[x+1]) + 0.5f * absf(r_m1[x-1] - r_m1[x+1]) +
                         0.5f * absf(r_p1[x-2] - r_p1[x]) + 0.5f * absf(r_m1[x-2] - r_m1[x]);

                // NE, SE, NW, SW 对于 Green 像素来说更简单（对角线是同一种异色）
                gra[4] = absf(r_m1[x+1]-r_p1[x-1]) + absf(r_m2[x+2]-val) + absf(r_m2[x+1]-r_0[x-1]) + absf(r_m1[x+2]-r_p1[x]);
                gra[5] = absf(r_p1[x+1]-r_m1[x-1]) + absf(r_p2[x+2]-val) + absf(r_p1[x+2]-r_m1[x]) + absf(r_p2[x+1]-r_0[x-1]);
                gra[6] = absf(r_m1[x-1]-r_p1[x+1]) + absf(r_m2[x-2]-val) + absf(r_m2[x-1]-r_0[x+1]) + absf(r_m1[x-2]-r_p1[x]);
                gra[7] = absf(r_p1[x-1]-r_m1[x+1]) + absf(r_p2[x-2]-val) + absf(r_p2[x-1]-r_0[x+1]) + absf(r_p1[x-2]-r_m1[x]);
            }

            // ================= 阈值选择 =================
            float T = get_threshold(gra);
            
            float Rsum = 0, Gsum = 0, Bsum = 0;
            int count = 0;

            // ================= 插值计算 =================
            // 这是一个小型的 Switch 结构，遍历 8 个方向
            for (int k = 0; k < 8; ++k) {
                // 梯度太大
                if (gra[k] >= T) continue;

                float r=0, g=0, b=0;

                // 核心逻辑：根据当前像素类型和方向 k，决定如何取平均
                // BGGR:
                // Red: (1,1). Neighbors N/S/E/W=G, Diag=B
                // Blue: (0,0). Neighbors N/S/E/W=G, Diag=R
                
                if (is_red) {
                    // R 在中心。需恢复 G 和 B。
                    // 逻辑与原代码 "Consider at Red pixels" 对应
                    switch(k) {
                        case 0: r=0.5f*(val+r_m2[x]); g=r_m1[x]; b=0.5f*(r_m1[x-1]+r_m1[x+1]); break; // N
                        case 1: r=0.5f*(val+r_0[x+2]); g=r_0[x+1]; b=0.5f*(r_m1[x+1]+r_p1[x+1]); break; // E
                        case 2: r=0.5f*(val+r_p2[x]); g=r_p1[x]; b=0.5f*(r_p1[x-1]+r_p1[x+1]); break; // S
                        case 3: r=0.5f*(val+r_0[x-2]); g=r_0[x-1]; b=0.5f*(r_m1[x-1]+r_p1[x-1]); break; // W
                        case 4: r=0.5f*(val+r_m2[x+2]); g=0.25f*(r_0[x+1]+r_m1[x+2]+r_m1[x]+r_m2[x+1]); b=r_m1[x+1]; break; // NE
                        case 5: r=0.5f*(val+r_p2[x+2]); g=0.25f*(r_0[x+1]+r_p1[x+2]+r_p1[x]+r_p2[x+1]); b=r_p1[x+1]; break; // SE
                        case 6: r=0.5f*(val+r_m2[x-2]); g=0.25f*(r_0[x-1]+r_m1[x-2]+r_m1[x]+r_m2[x-1]); b=r_m1[x-1]; break; // NW
                        case 7: r=0.5f*(val+r_p2[x-2]); g=0.25f*(r_0[x-1]+r_p1[x-2]+r_p1[x]+r_p2[x-1]); b=r_p1[x-1]; break; // SW
                    }
                } 
                else if (is_blue) {
                    // B 在中心。需恢复 G 和 R。
                    // 逻辑与 "Consider at Red pixels" 对称，只是交换 R 和 B 的位置
                    switch(k) {
                        case 0: b=0.5f*(val+r_m2[x]); g=r_m1[x]; r=0.5f*(r_m1[x-1]+r_m1[x+1]); break;
                        case 1: b=0.5f*(val+r_0[x+2]); g=r_0[x+1]; r=0.5f*(r_m1[x+1]+r_p1[x+1]); break;
                        case 2: b=0.5f*(val+r_p2[x]); g=r_p1[x]; r=0.5f*(r_p1[x-1]+r_p1[x+1]); break;
                        case 3: b=0.5f*(val+r_0[x-2]); g=r_0[x-1]; r=0.5f*(r_m1[x-1]+r_p1[x-1]); break;
                        case 4: b=0.5f*(val+r_m2[x+2]); g=0.25f*(r_0[x+1]+r_m1[x+2]+r_m1[x]+r_m2[x+1]); r=r_m1[x+1]; break;
                        case 5: b=0.5f*(val+r_p2[x+2]); g=0.25f*(r_0[x+1]+r_p1[x+2]+r_p1[x]+r_p2[x+1]); r=r_p1[x+1]; break;
                        case 6: b=0.5f*(val+r_m2[x-2]); g=0.25f*(r_0[x-1]+r_m1[x-2]+r_m1[x]+r_m2[x-1]); r=r_m1[x-1]; break;
                        case 7: b=0.5f*(val+r_p2[x-2]); g=0.25f*(r_0[x-1]+r_p1[x-2]+r_p1[x]+r_p2[x-1]); r=r_p1[x-1]; break;
                    }
                }
                else if (is_green_b) {
                    // G 在中心 (BGGR中的 (0,1), 偶行奇列)。行是 B G B G，列是 G R G R。
                    // 所以：左右是 B (X轴)，上下是 R (Y轴)。
                    // 对应原代码： "consider those green pixels at the lower-right corner" (G1y)
                    // 原代码 G1y 的逻辑是 "Vertically R, Horizontally B"。这正是 BGGR (0,1) 的情况。
                    switch(k) {
                        case 0: g=0.5f*(val+r_m2[x]); r=r_m1[x]; b=0.25f*(r_m2[x-1]+r_m2[x+1]+r_0[x-1]+r_0[x+1]); break;
                        case 1: g=0.5f*(val+r_0[x+2]); b=r_0[x+1]; r=0.25f*(r_m1[x]+r_p1[x]+r_m1[x+2]+r_p1[x+2]); break;
                        case 2: g=0.5f*(val+r_p2[x]); r=r_p1[x]; b=0.25f*(r_0[x-1]+r_0[x+1]+r_p2[x-1]+r_p2[x+1]); break;
                        case 3: g=0.5f*(val+r_0[x-2]); b=r_0[x-1]; r=0.25f*(r_m1[x-2]+r_m1[x]+r_p1[x-2]+r_p1[x]); break;
                        case 4: b=0.5f*(r_m2[x+1]+r_0[x+1]); r=0.5f*(r_m1[x]+r_m1[x+2]); g=r_m1[x+1]; break;
                        case 5: b=0.5f*(r_0[x+1]+r_p2[x+1]); r=0.5f*(r_p1[x]+r_p1[x+2]); g=r_p1[x+1]; break;
                        case 6: b=0.5f*(r_0[x-1]+r_m2[x-1]); r=0.5f*(r_m1[x-2]+r_m1[x]); g=r_m1[x-1]; break;
                        case 7: b=0.5f*(r_0[x-1]+r_p2[x-1]); r=0.5f*(r_p1[x-2]+r_p1[x]); g=r_p1[x-1]; break;
                    }
                }
                else { // is_green_r
                    // G 在中心 (BGGR中的 (1,0), 奇行偶列)。行是 G R G R，列是 B G B G。
                    // 所以：左右是 R (X轴)，上下是 B (Y轴)。
                    // 对应原代码： "consider those green pixels at upper-left" (G0y)
                    // 原代码 G0y 的逻辑是 "Vertically B, Horizontally R"。
                    switch(k) {
                        case 0: g=0.5f*(val+r_m2[x]); b=r_m1[x]; r=0.25f*(r_m2[x-1]+r_m2[x+1]+r_0[x-1]+r_0[x+1]); break;
                        case 1: g=0.5f*(val+r_0[x+2]); r=r_0[x+1]; b=0.25f*(r_m1[x]+r_p1[x]+r_m1[x+2]+r_p1[x+2]); break;
                        case 2: g=0.5f*(val+r_p2[x]); b=r_p1[x]; r=0.25f*(r_0[x-1]+r_0[x+1]+r_p2[x-1]+r_p2[x+1]); break;
                        case 3: g=0.5f*(val+r_0[x-2]); r=r_0[x-1]; b=0.25f*(r_m1[x-2]+r_m1[x]+r_p1[x-2]+r_p1[x]); break;
                        case 4: r=0.5f*(r_m2[x+1]+r_0[x+1]); b=0.5f*(r_m1[x]+r_m1[x+2]); g=r_m1[x+1]; break;
                        case 5: r=0.5f*(r_0[x+1]+r_p2[x+1]); b=0.5f*(r_p1[x]+r_p1[x+2]); g=r_p1[x+1]; break;
                        case 6: r=0.5f*(r_0[x-1]+r_m2[x-1]); b=0.5f*(r_m1[x-2]+r_m1[x]); g=r_m1[x-1]; break;
                        case 7: r=0.5f*(r_0[x-1]+r_p2[x-1]); b=0.5f*(r_p1[x-2]+r_p1[x]); g=r_p1[x-1]; break;
                    }
                }
                
                Rsum += r; Gsum += g; Bsum += b;
                count++;
            } // end 8 directions loop

            // ================= 最终赋值 =================
            if (count > 0) {
                float inv_cnt = 1.0f / count;
                // VNG 色差恒定原理: OutColor = BaseColor + (AvgDiff)
                // 原代码: outG = inR + (Gsum - Rsum)/cnt
                // 变换为: outG = inR - Rsum/cnt + Gsum/cnt
                // 但注意，原代码里的 sum 其实是 Color Average (Rave, Gave...)
                // Rsum = sum(Rave)
                
                // Red Pixel: outG = val + (Gsum - Rsum)/cnt; outB = val + (Bsum - Rsum)/cnt; outR = val
                // Blue Pixel: outG = val + (Gsum - Bsum)/cnt; outR = val + (Rsum - Bsum)/cnt; outB = val
                // Green Pixel: outR = val + (Rsum - Gsum)/cnt; outB = val + (Bsum - Gsum)/cnt; outG = val

                float diff = 0;
                if (is_red) {
                    diff = (Gsum - Rsum) * inv_cnt;
                    out_ptr[x][1] = val + diff; // G
                    diff = (Bsum - Rsum) * inv_cnt;
                    out_ptr[x][0] = val + diff; // B
                    out_ptr[x][2] = val;        // R
                } else if (is_blue) {
                    diff = (Gsum - Bsum) * inv_cnt;
                    out_ptr[x][1] = val + diff; // G
                    diff = (Rsum - Bsum) * inv_cnt;
                    out_ptr[x][2] = val + diff; // R
                    out_ptr[x][0] = val;        // B
                } else { // Green
                    diff = (Rsum - Gsum) * inv_cnt;
                    out_ptr[x][2] = val + diff; // R
                    diff = (Bsum - Gsum) * inv_cnt;
                    out_ptr[x][0] = val + diff; // B
                    out_ptr[x][1] = val;        // G
                }
            } else {
                // 如果没有方向满足阈值（全平滑？全噪点？），回退到最近邻或保持原色
                // 原代码逻辑：直接赋值原色
                if (is_red) { out_ptr[x][2] = val; out_ptr[x][1] = val; out_ptr[x][0] = val; } // 这样会变灰度，原代码其实只是没加 diff
                else if (is_blue) { out_ptr[x][0] = val; out_ptr[x][1] = val; out_ptr[x][2] = val; }
                else { out_ptr[x][1] = val; out_ptr[x][0] = val; out_ptr[x][2] = val; }
            }
            
            // 简单的防止下溢出（VNG可能产生负值）
            if (out_ptr[x][0] < 0) out_ptr[x][0] = 0;
            if (out_ptr[x][1] < 0) out_ptr[x][1] = 0;
            if (out_ptr[x][2] < 0) out_ptr[x][2] = 0;
        }
    }

    // 4) 将 Float 结果转回原始深度 (8位或16位)，保持 3 通道
    out_f.convertTo(dst, CV_MAKETYPE(src.depth(), 3));
}

} // namespace

// 对外接口：保持项目统一命名
void demosiacVNG(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    CV_Assert(!raw.empty());
    CV_Assert(raw.channels() == 1);
    CV_Assert(pattern == BGGR && "Current optimized VNG only supports BGGR.");

    // 先用一个“边界稳定”的 demosaic 作为兜底（本项目自带 HA-style demosaic，边界用 REFLECT_101 pad）
    // 再用 VNG 的结果覆盖中间区域，避免 VNG 在边缘 2px 没有 5x5 窗口导致的“边角发灰/颜色混在一起”。
    cv::Mat base;
    demosiac(raw, base, pattern); // CV_16UC3 / CV_8UC3

    cv::Mat vng;
    vng_demosaic_bggr(raw, vng);

    // 用 VNG 覆盖 interior（忽略边缘 2 像素）
    base.copyTo(dst);
    const int y0 = 2;
    const int y1 = raw.rows - 2;
    const int x0 = 2;
    const int x1 = raw.cols - 2;

    for (int y = y0; y < y1; ++y) {
        const uint16_t* vrow16 = (vng.depth() == CV_16U) ? vng.ptr<uint16_t>(y) : nullptr;
        uint16_t* drow16 = (dst.depth() == CV_16U) ? dst.ptr<uint16_t>(y) : nullptr;
        const uint8_t* vrow8 = (vng.depth() == CV_8U) ? vng.ptr<uint8_t>(y) : nullptr;
        uint8_t* drow8 = (dst.depth() == CV_8U) ? dst.ptr<uint8_t>(y) : nullptr;

        if (vrow16 && drow16) {
            for (int x = x0; x < x1; ++x) {
                drow16[x * 3 + 0] = vrow16[x * 3 + 0];
                drow16[x * 3 + 1] = vrow16[x * 3 + 1];
                drow16[x * 3 + 2] = vrow16[x * 3 + 2];
            }
        } else if (vrow8 && drow8) {
            for (int x = x0; x < x1; ++x) {
                drow8[x * 3 + 0] = vrow8[x * 3 + 0];
                drow8[x * 3 + 1] = vrow8[x * 3 + 1];
                drow8[x * 3 + 2] = vrow8[x * 3 + 2];
            }
        } else {
            // 不应发生：vng/base/dst 位深应一致
            CV_Assert(false && "Unexpected depth mismatch in demosiacVNG");
        }
    }
}