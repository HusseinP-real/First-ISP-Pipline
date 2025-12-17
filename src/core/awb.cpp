#include "awb.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void runAWB(cv::Mat& rawImage, AWBGains& gains, bool enableAuto) {
    if (rawImage.empty()) return;

    
    // statistic
    if (enableAuto) {
        long long sumR = 0, sumG = 0, sumB = 0;
        int countR = 0, countG = 0, countB = 0;

        for (int y = 0; y < rawImage.rows; y++) {
            const uint16_t* row = rawImage.ptr<uint16_t>(y);
            for (int x = 0; x < rawImage.cols; x++) {

                uint16_t value = row[x];

                // filter
                //if (value < 20 || value > 240) continue;
                if (value < 100 || value > 60000) continue;

                // judge rgb
                if (y % 2 == 0) {
                    // r g r g ...
                    if (x % 2 == 0) {
                        sumR += value;
                        countR++;
                    } else {
                        sumG += value;
                        countG++;
                    }
                } else {
                    // g b g b ...
                    if (x % 2 == 0) {
                        sumG += value;
                        countG++;
                    } else {
                        sumB += value;
                        countB++;
                    }
                    
                }
            }
        }

        // calculate gains
        if (countR != 0 && countG != 0 && countB != 0) {
        
            float meanR = static_cast<float>(sumR) / countR;
            float meanG = static_cast<float>(sumG) / countG;
            float meanB = static_cast<float>(sumB) / countB;

            gains.r = meanG / meanR;
            gains.g = 1.0f;
            gains.b = meanG / meanB;

            std::cout << "Valid pixels found, calculated gains: R=" << gains.r << ", G=" << gains.g << ", B=" << gains.b << std::endl;


        } else {
            gains.r = 1.0f;
            gains.g = 1.0f;
            gains.b = 1.0f;

            std::cout << "No valid pixels found, using default gains" << std::endl;
        }

    } else {
        std::cout << "[manual AWB]Valid pixels found, calculated gains: R=" << gains.r << ", G=" << gains.g << ", B=" << gains.b << std::endl;
    }
    

    // apply gains
    for (int y = 0; y < rawImage.rows; y++) {
        uint16_t* row = rawImage.ptr<uint16_t>(y);
        for (int x = 0; x < rawImage.cols; x++) {
            uint16_t value = row[x];
            float gain = 1.0f;

            // rggb pattern determine
            if (y % 2 == 0) {
                if (x % 2 == 0) {
                    gain = gains.r;
                } else {
                    gain = gains.g;
                }
            } else {
                if (x % 2 == 0) {
                    gain = gains.g;
                } else {
                    gain = gains.b;
                }
            }

            // apply gains
            row[x] = cv::saturate_cast<uint16_t>(value * gain);
        }
    }
}
