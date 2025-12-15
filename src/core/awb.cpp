#include "awb.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void runAWB(cv::Mat& rawImage, AWBGains& gains, bool enableAuto) {
    if (rawImage.empty()) return;

    

    // statistic
    if (enableAuto) {
        long long sumR = 0, sumG = 0, sumB = 0;
        int validPixelCount = 0;

        for (int y = 0; y < rawImage.rows; y++) {
            const uchar* row = rawImage.ptr<uchar>(y);
            for (int x = 0; x < rawImage.cols; x++) {
                // B G R
                uchar b = row[x * 3 + 0];
                uchar g = row[x * 3 + 1];
                uchar r = row[x * 3 + 2];

                // filter

                // too dark
                if (b < 20 || g < 20 || r < 20) continue;

                // too bright
                if (b > 240 || g > 240 || r > 240) continue;

                // optional middle gray filter

                sumR += r;
                sumG += g;
                sumB += b;
                validPixelCount++;
            }
        }

        // calculate gains
        
        if (validPixelCount == 0) {
            gains.r = 1.0f;
            gains.g = 1.0f;
            gains.b = 1.0f;

            std::cout << "No valid pixels found, using default gains" << std::endl;
        
        } else {
            float meanR = static_cast<float>(sumR) / validPixelCount;
            float meanG = static_cast<float>(sumG) / validPixelCount;
            float meanB = static_cast<float>(sumB) / validPixelCount;

            gains.r = meanG / meanR;
            gains.g = 1.0f;
            gains.b = meanG / meanB;

            std::cout << "Valid pixels found, calculated gains: R=" << gains.r << ", G=" << gains.g << ", B=" << gains.b << std::endl;


        }

    } else {
        std::cout << "[manual AWB]Valid pixels found, calculated gains: R=" << gains.r << ", G=" << gains.g << ", B=" << gains.b << std::endl;
    }
    

    // apply gains
    for (int y = 0; y < rawImage.rows; y++) {
        uchar* row = rawImage.ptr<uchar>(y);
        for (int x = 0; x < rawImage.cols; x++) {
            // B G R
            uchar b = row[x * 3 + 0];
            uchar g = row[x * 3 + 1];
            uchar r = row[x * 3 + 2];

            // apply gains
            row[x * 3 + 0] = cv::saturate_cast<uchar>(b * gains.b);
            row[x * 3 + 1] = cv::saturate_cast<uchar>(g * gains.g);
            row[x * 3 + 2] = cv::saturate_cast<uchar>(r * gains.r);
        }
    }
}
