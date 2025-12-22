#include "../core/raw_reader.h"
#include "../core/blc.h"
#include "../core/denoise.h"
#include "../core/awb.h"
#include "../core/gamma.h"
#include "../core/vgn.h"
#include "../core/ccm.h"
#include "../core/sharpen.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // 基础参数
    const std::string inputFile = "data/input/raw1.raw";
    const int width = 512;
    const int height = 500;
    const int frameIndex = 0;

    // 1) 读取 RAW (16-bit 单通道，BGGR)
    cv::Mat raw = readRawToMat(inputFile, width, height, frameIndex);
    if (raw.empty()) {
        std::cerr << "Failed to read RAW image: " << inputFile << std::endl;
        return -1;
    }
    if (raw.type() != CV_16UC1) {
        std::cerr << "Error: Expected CV_16UC1 raw input." << std::endl;
        return -1;
    }
    std::cout << "Raw loaded. size=" << raw.cols << "x" << raw.rows << std::endl;

    // 2) 黑电平校正 BLC
    imageInfo info{width, height, 0};
    blackLevels bls{200.f, 200.f, 200.f, 200.f};
    applyBlc(reinterpret_cast<uint16_t*>(raw.data), info, bls);
    std::cout << "BLC applied." << std::endl;

    // 3) Denoise (降噪) - 在RAW域进行，BLC之后、AWB之前
    std::cout << "Applying Denoise..." << std::endl;
    std::vector<uint16_t> raw_vector(width * height);
    std::vector<uint16_t> raw_denoised;

    const uint16_t* raw_data = raw.ptr<uint16_t>(0);
    std::copy(raw_data, raw_data + width * height, raw_vector.begin());

    runDenoise(raw_vector, raw_denoised, width, height);

    uint16_t* raw_output_data = raw.ptr<uint16_t>(0);
    std::copy(raw_denoised.begin(), raw_denoised.end(), raw_output_data);
    std::cout << "Denoise applied." << std::endl;

    // 4) AWB
    AWBGains gains{1.4f, 1.0f, 1.2f};
    runAWB(raw, gains, false);
    std::cout << "[Manual AWB] Gains: R=" << gains.r << " G=" << gains.g << " B=" << gains.b << std::endl;

    // 5) Demosaic (VNG, BGGR -> BGR, 保持 16-bit)
    cv::Mat color16;
    demosiacVNG(raw, color16, BGGR);
    std::cout << "Demosaic done (VNG)." << std::endl;

    // 6) CCM
    std::cout << "Applying Color Correction Matrix (CCM)..." << std::endl;
    float ccm_matrix[3][3] = {
        {1.546875f, -0.29296875f, -0.15625f},
        {-0.3125f, 1.71484375f, -0.40625f},
        {-0.2265625f, -0.3515625f, 1.609375f}
    };

    ColorCorrectionMatrix ccm(ccm_matrix, 16);

    std::vector<uint16_t> src_rgb;
    std::vector<uint16_t> dst_rgb;
    int pixel_count = color16.rows * color16.cols;
    src_rgb.reserve(pixel_count * 3);

    for (int y = 0; y < color16.rows; y++) {
        const uint16_t* row = color16.ptr<uint16_t>(y);
        for (int x = 0; x < color16.cols; x++) {
            src_rgb.push_back(row[3 * x + 2]); // R
            src_rgb.push_back(row[3 * x + 1]); // G
            src_rgb.push_back(row[3 * x + 0]); // B
        }
    }

    ccm.process(src_rgb, dst_rgb);
    std::cout << "CCM applied. Processed " << pixel_count << " pixels." << std::endl;

    cv::Mat color16_ccm = cv::Mat::zeros(color16.rows, color16.cols, CV_16UC3);
    for (int y = 0; y < color16_ccm.rows; y++) {
        uint16_t* row = color16_ccm.ptr<uint16_t>(y);
        for (int x = 0; x < color16_ccm.cols; x++) {
            int idx = (y * color16_ccm.cols + x) * 3;
            row[3 * x + 0] = dst_rgb[idx + 2]; // B
            row[3 * x + 1] = dst_rgb[idx + 1]; // G
            row[3 * x + 2] = dst_rgb[idx + 0]; // R
        }
    }

    // 7) Digital Gain
    double gain = 8.0;
    cv::Mat color16_gain;
    color16_ccm.convertTo(color16_gain, CV_16UC3, gain);
    std::cout << "Digital gain applied (16-bit): " << gain << "x" << std::endl;

    // 8) Gamma with dithering
    GammaCorrection gamma;
    cv::Mat color8_gamma;
    gamma.runWithDithering(color16_gain, color8_gamma);
    if (color8_gamma.empty()) {
        std::cerr << "Gamma output is empty." << std::endl;
        return -1;
    }
    std::cout << "Gamma applied with dithering (16-bit -> 8-bit + gamma=2.4)." << std::endl;

    // // 9) Sharpen
    // std::cout << "Applying Sharpen (post-gamma)..." << std::endl;
    // cv::Mat color16_gamma_for_sharpen;
    // color8_gamma.convertTo(color16_gamma_for_sharpen, CV_16UC3, 65535.0 / 255.0);
    // sharpen(color16_gamma_for_sharpen, 1.0f, 1, 1280);
    // cv::Mat color8_gamma_sharpen;
    // color16_gamma_for_sharpen.convertTo(color8_gamma_sharpen, CV_8UC3, 255.0 / 65535.0);
    // std::cout << "Sharpen applied." << std::endl;

    // 10) 输出
    std::string outFile = "data/output/raw1_pipeline_vng_gamma.png";
    std::string outFileSharpened = "data/output/raw1_pipeline_vng_gamma_sharpened.png";

    if (cv::imwrite(outFile, color8_gamma)) {
        std::cout << "Saved (VNG + gamma, 8-bit): " << outFile << std::endl;
    } else {
        std::cerr << "Failed to save: " << outFile << std::endl;
    }

    // if (cv::imwrite(outFileSharpened, color8_gamma_sharpen)) {
    //     std::cout << "Saved (VNG + gamma + sharpen, 8-bit): " << outFileSharpened << std::endl;
    // } else {
    //     std::cerr << "Failed to save: " << outFileSharpened << std::endl;
    //     return -1;
    // }

    return 0;
}


