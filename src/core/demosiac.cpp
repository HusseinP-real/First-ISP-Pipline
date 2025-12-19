#include "demosiac.h" // 建议重命名为 demosaic.h
#include <opencv2/opencv.hpp>
#include <algorithm>

// 使用 saturate_cast 替代宏，更安全且利用 OpenCV 优化
// 或者保留你的宏，但建议加上 const 和括号保护
template<typename T>
inline T clip_val(int val) {
    return cv::saturate_cast<T>(val);
}

void demosiac(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    // 1. 边界填充：使用 REFLECT_101 保持 Bayer 阵列的奇偶性
    cv::Mat padded_raw;
    cv::copyMakeBorder(raw, padded_raw, 2, 2, 2, 2, cv::BORDER_REFLECT_101);

    // Two-pass demosaic:
    // 1) reconstruct full-resolution Green with edge-aware interpolation
    // 2) reconstruct Red/Blue using color-difference (R-G, B-G) interpolation
    //
    // This significantly reduces zippering / false color compared to pure bilinear R/B.
    cv::Mat G(raw.rows, raw.cols, CV_16UC1);

    int height = raw.rows;
    int width = raw.cols;

    int start_y = 0;
    int start_x = 0;

    // 设置起始偏移
    switch (pattern) {
        case RGGB: start_y = 0; start_x = 0; break;
        case BGGR: start_y = 1; start_x = 1; break;
        case GBRG: start_y = 1; start_x = 0; break;
        case GRBG: start_y = 0; start_x = 1; break;
    }

    // ---- Pass 1: Green plane ----
    for (int y = 0; y < height; y++) {
        int py = y + 2;

        const uint16_t* r_m2 = padded_raw.ptr<uint16_t>(py - 2);
        const uint16_t* r_m1 = padded_raw.ptr<uint16_t>(py - 1);
        const uint16_t* r_cur = padded_raw.ptr<uint16_t>(py);
        const uint16_t* r_p1 = padded_raw.ptr<uint16_t>(py + 1);
        const uint16_t* r_p2 = padded_raw.ptr<uint16_t>(py + 2);

        uint16_t* gRow = G.ptr<uint16_t>(y);

        // 优化：将行奇偶性判断移出内层循环
        bool ye = ((y + start_y) & 1) == 0; 
        
        // 优化：计算行的列起始奇偶性
        int x_parity_start = (start_x) & 1;

        for (int x = 0; x < width; x++) {
            int px = x + 2;
            
            // 优化：不再每次计算 ((x+start_x)&1)，而是利用 x 的奇偶性翻转
            // xe 为 true 代表列坐标（加上偏移后）是偶数
            bool xe = ((x & 1) ^ x_parity_start) == 0;

            // --- G Channel 处理 ---
            // 逻辑简化：如果 (行偶且列奇) 或者 (行奇且列偶)，则是 G 像素
            // 等价于 ye != xe
            if (ye != xe) {
                // 当前是 G 像素，直接拷贝
                gRow[x] = r_cur[px];
            } else {
                // 当前是 R 或 B 像素，需要插值 G
                // 使用简单的 Hamilton-Adams 风格梯度判断
                int grad_h = std::abs((int)r_cur[px - 2] - (int)r_cur[px + 2]);
                int grad_v = std::abs((int)r_m2[px] - (int)r_p2[px]);

                if (grad_h < grad_v) {
                    gRow[x] = static_cast<uint16_t>((r_cur[px - 1] + r_cur[px + 1] + 1) / 2); // +1 用于四舍五入
                } else if (grad_v < grad_h) {
                    gRow[x] = static_cast<uint16_t>((r_m1[px] + r_p1[px] + 1) / 2);
                } else {
                    gRow[x] = static_cast<uint16_t>((r_cur[px - 1] + r_cur[px + 1] + r_m1[px] + r_p1[px] + 2) / 4);
                }
            }
        }
    }

    // Pad G for easy neighbor access consistent with padded_raw indexing
    cv::Mat padded_G;
    cv::copyMakeBorder(G, padded_G, 2, 2, 2, 2, cv::BORDER_REFLECT_101);

    // ---- Pass 2: Red/Blue via color-difference interpolation ----
    dst = cv::Mat(raw.rows, raw.cols, CV_16UC3);
    for (int y = 0; y < height; y++) {
        int py = y + 2;

        const uint16_t* r_m1 = padded_raw.ptr<uint16_t>(py - 1);
        const uint16_t* r_cur = padded_raw.ptr<uint16_t>(py);
        const uint16_t* r_p1 = padded_raw.ptr<uint16_t>(py + 1);

        const uint16_t* g_m1 = padded_G.ptr<uint16_t>(py - 1);
        const uint16_t* g_cur = padded_G.ptr<uint16_t>(py);
        const uint16_t* g_p1 = padded_G.ptr<uint16_t>(py + 1);

        cv::Vec3w* p_dst = dst.ptr<cv::Vec3w>(y);

        bool ye = ((y + start_y) & 1) == 0;
        int x_parity_start = (start_x) & 1;

        for (int x = 0; x < width; x++) {
            int px = x + 2;
            bool xe = ((x & 1) ^ x_parity_start) == 0;

            int g_val = static_cast<int>(g_cur[px]);
            int r_val = 0;
            int b_val = 0;

            if (ye && xe) {
                // Red site: R known, B from diagonal (B-G) + G
                r_val = static_cast<int>(r_cur[px]);
                int d1 = static_cast<int>(r_m1[px - 1]) - static_cast<int>(g_m1[px - 1]);
                int d2 = static_cast<int>(r_m1[px + 1]) - static_cast<int>(g_m1[px + 1]);
                int d3 = static_cast<int>(r_p1[px - 1]) - static_cast<int>(g_p1[px - 1]);
                int d4 = static_cast<int>(r_p1[px + 1]) - static_cast<int>(g_p1[px + 1]);
                b_val = g_val + (d1 + d2 + d3 + d4 + 2) / 4;
            } else if (!ye && !xe) {
                // Blue site: B known, R from diagonal (R-G) + G
                b_val = static_cast<int>(r_cur[px]);
                int d1 = static_cast<int>(r_m1[px - 1]) - static_cast<int>(g_m1[px - 1]);
                int d2 = static_cast<int>(r_m1[px + 1]) - static_cast<int>(g_m1[px + 1]);
                int d3 = static_cast<int>(r_p1[px - 1]) - static_cast<int>(g_p1[px - 1]);
                int d4 = static_cast<int>(r_p1[px + 1]) - static_cast<int>(g_p1[px + 1]);
                r_val = g_val + (d1 + d2 + d3 + d4 + 2) / 4;
            } else if (ye && !xe) {
                // Green on Red row: R from left/right (R-G), B from up/down (B-G)
                b_val = g_val + ((static_cast<int>(r_m1[px]) - static_cast<int>(g_m1[px])) +
                                 (static_cast<int>(r_p1[px]) - static_cast<int>(g_p1[px])) + 1) / 2;
                r_val = g_val + ((static_cast<int>(r_cur[px - 1]) - static_cast<int>(g_cur[px - 1])) +
                                 (static_cast<int>(r_cur[px + 1]) - static_cast<int>(g_cur[px + 1])) + 1) / 2;
            } else {
                // Green on Blue row: B from left/right (B-G), R from up/down (R-G)
                b_val = g_val + ((static_cast<int>(r_cur[px - 1]) - static_cast<int>(g_cur[px - 1])) +
                                 (static_cast<int>(r_cur[px + 1]) - static_cast<int>(g_cur[px + 1])) + 1) / 2;
                r_val = g_val + ((static_cast<int>(r_m1[px]) - static_cast<int>(g_m1[px])) +
                                 (static_cast<int>(r_p1[px]) - static_cast<int>(g_p1[px])) + 1) / 2;
            }

            p_dst[x][0] = clip_val<uint16_t>(b_val);
            p_dst[x][1] = clip_val<uint16_t>(g_val);
            p_dst[x][2] = clip_val<uint16_t>(r_val);
        }
    }
}