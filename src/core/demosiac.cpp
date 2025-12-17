#include "demosiac.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void demosiac(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    // 初始化输出图像为 16-bit 三通道 BGR
    dst = cv::Mat::zeros(raw.rows, raw.cols, CV_16UC3);

    int height = raw.rows;
    int width = raw.cols;
    
    if (pattern != RGGB) {
        std::cerr << "Warning: Only RGGB pattern is currently implemented." << std::endl;
    }

    for (int y = 1; y < height - 1; y++) {
        const uint16_t* row = raw.ptr<uint16_t>(y);
        const uint16_t* row_up = raw.ptr<uint16_t>(y - 1);
        const uint16_t* row_down = raw.ptr<uint16_t>(y + 1);

        uint16_t* p_dst = dst.ptr<uint16_t>(y);

        for (int x = 1; x < width - 1; x++) {
            uint16_t b_val = 0, g_val = 0, r_val = 0;

            bool ye = (y & 1) == 0;  // y is even
            bool xe = (x & 1) == 0;  // x is even

            // RGGB pattern layout:
            // Even rows: R G R G ...
            // Odd rows:  G B G B ...
            if (ye && xe) {
                // Even row, even col: R pixel
                r_val = row[x];
                g_val = (row[x - 1] + row[x + 1] + row_up[x] + row_down[x]) / 4;
                b_val = (row_up[x - 1] + row_up[x + 1] + row_down[x - 1] + row_down[x + 1]) / 4;
            } else if (ye && !xe) {
                // Even row, odd col: G pixel
                r_val = (row[x - 1] + row[x + 1]) / 2;
                g_val = row[x];
                b_val = (row_up[x] + row_down[x]) / 2;
            } else if (!ye && xe) {
                // Odd row, even col: G pixel
                r_val = (row_up[x] + row_down[x]) / 2;
                g_val = row[x];
                b_val = (row[x - 1] + row[x + 1]) / 2;
            } else {
                // Odd row, odd col: B pixel
                r_val = (row_up[x - 1] + row_up[x + 1] + row_down[x - 1] + row_down[x + 1]) / 4;
                g_val = (row[x - 1] + row[x + 1] + row_up[x] + row_down[x]) / 4;
                b_val = row[x];
            }

            // output to dst (BGR format: OpenCV standard)
            p_dst[3*x + 0] = b_val;  // B
            p_dst[3*x + 1] = g_val;  // G
            p_dst[3*x + 2] = r_val;  // R
        }
    }
}