/**
 * @file main.cpp
 * @brief BM3D Color Image Denoising (Clean OpenCV Version)
 * * 修正版说明：
 * 1. 移除了所有自定义的 converter.h 依赖，解决色彩空间不匹配导致的彩噪问题。
 * 2. 使用 cv::cvtColor 进行标准化的色彩空间转换。
 * 3. 色度通道使用强力高斯模糊 (15x15) 去除色斑。
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include "include/bm3d_denoiser.hpp"

// 注意：这里不再需要 include "../src/core/bmp/converter.h"

namespace fs = std::filesystem;

// ==================== 辅助函数 ====================

std::vector<std::string> getBMPFiles(const std::string& directory) {
    std::vector<std::string> bmpFiles;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        std::cerr << "Error: Could not open directory: " << directory << std::endl;
        return bmpFiles;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.length() > 4) {
            std::string ext = filename.substr(filename.length() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".bmp") {
                bmpFiles.push_back(directory + "/" + filename);
            }
        }
    }
    closedir(dir);
    std::sort(bmpFiles.begin(), bmpFiles.end());
    return bmpFiles;
}

cv::Mat addGaussianNoise(const cv::Mat& img, double sigma) {
    cv::Mat noisy;
    img.convertTo(noisy, CV_64F);
    cv::Mat noise(noisy.size(), CV_64FC3);
    cv::randn(noise, cv::Scalar(0, 0, 0), cv::Scalar(sigma, sigma, sigma));
    noisy += noise;
    return noisy;
}

double computePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat i1, i2;
    img1.convertTo(i1, CV_64F);
    img2.convertTo(i2, CV_64F);
    cv::Mat diff = i1 - i2;
    cv::Scalar mseScalar = cv::mean(diff.mul(diff));
    double mse = (mseScalar[0] + mseScalar[1] + mseScalar[2]) / 3.0;
    if (mse == 0) return std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(255.0 / std::sqrt(mse));
}

// ==================== 核心降噪逻辑 (全 OpenCV 版) ====================

cv::Mat denoiseColorImage(const cv::Mat& noisyBGR, float sigma) {
    // 步骤 1: 转换到 8位 并截断 (Saturate)
    // 这一步非常重要，它去除了因为加噪声产生的越界值(<0 或 >255)，恢复正常的黑电平
    cv::Mat noisyBGR_8u;
    noisyBGR.convertTo(noisyBGR_8u, CV_8UC3);
    
    // 步骤 2: 转换颜色空间 BGR -> YCrCb (使用 OpenCV 标准)
    cv::Mat ycrcb;
    cv::cvtColor(noisyBGR_8u, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // 步骤 3: 分离通道
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    // channels[0] = Y (Luminance)
    // channels[1] = Cr (Red Chroma)
    // channels[2] = Cb (Blue Chroma)
    
    // 步骤 4: Y 通道应用 BM3D (保持纹理细节)
    std::cout << "  [Y Channel] Applying BM3D..." << std::flush;
    auto start_y = std::chrono::high_resolution_clock::now();
    
    BM3D_Denoiser denoiser;
    // 修正：Y 通道的噪声标准差小于原始 RGB 噪声
    // Y = 0.299R + 0.587G + 0.114B
    // Sigma_Y = sqrt(0.299^2 + 0.587^2 + 0.114^2) * Sigma_RGB ≈ 0.67 * Sigma_RGB
    float sigma_y = sigma * 0.67f;
    cv::Mat Y_denoised = denoiser.denoise(channels[0], sigma_y);
    
    // 确保类型一致
    if (Y_denoised.depth() != CV_8U) {
        Y_denoised.convertTo(channels[0], CV_8U);
    } else {
        Y_denoised.copyTo(channels[0]);
    }
    
    auto end_y = std::chrono::high_resolution_clock::now();
    std::cout << " done (" << std::chrono::duration_cast<std::chrono::milliseconds>(end_y - start_y).count() << " ms)" << std::endl;
    
    // 步骤 5: Cr/Cb 通道应用强力高斯模糊 (去除彩噪)
    std::cout << "  [Chroma] Applying Gaussian Blur..." << std::flush;
    
    // 15x15 的核足够抹平大部分色斑。如果还有，可以加大到 21x21。
    // sigmaX=0 表示让 OpenCV 自动根据核大小计算 sigma。
    cv::GaussianBlur(channels[1], channels[1], cv::Size(15, 15), 0);
    cv::GaussianBlur(channels[2], channels[2], cv::Size(15, 15), 0);
    
    std::cout << " done." << std::endl;
    
    // 步骤 6: 合并通道
    cv::Mat ycrcb_denoised;
    cv::merge(channels, ycrcb_denoised);
    
    // 步骤 7: 转回 BGR (OpenCV 标准逆转换，保证无色偏)
    cv::Mat bgr_denoised;
    cv::cvtColor(ycrcb_denoised, bgr_denoised, cv::COLOR_YCrCb2BGR);
    
    return bgr_denoised;
}

// ==================== 主程序流程 ====================

void processBMP(const std::string& bmpPath, const std::string& outputDir, float sigma) {
    std::string filename = bmpPath.substr(bmpPath.find_last_of("/\\") + 1);
    std::string baseName = filename.substr(0, filename.find_last_of("."));
    
    std::cout << "\nProcessing: " << filename << std::endl;
    
    // 读取图像
    cv::Mat original = cv::imread(bmpPath, cv::IMREAD_COLOR);
    if (original.empty()) return;
    
    // 加噪 (仅演示用)
    cv::Mat noisy = addGaussianNoise(original, sigma);
    
    // 保存噪点图
    cv::Mat noisyDisplay;
    noisy.convertTo(noisyDisplay, CV_8UC3);
    cv::imwrite(outputDir + "/" + baseName + "_noisy.png", noisyDisplay);
    
    double noisyPSNR = computePSNR(original, noisyDisplay);
    std::cout << "Noisy PSNR: " << std::fixed << std::setprecision(2) << noisyPSNR << " dB" << std::endl;
    
    // 执行降噪
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat denoised = denoiseColorImage(noisy, sigma);
    auto end = std::chrono::high_resolution_clock::now();
    
    double denoisedPSNR = computePSNR(original, denoised);
    std::cout << "Denoised PSNR: " << std::fixed << std::setprecision(2) << denoisedPSNR << " dB" << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;
    
    // 保存结果
    cv::imwrite(outputDir + "/" + baseName + "_bm3d.png", denoised);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./bm3d_denoise [input_dir] [sigma] [output_dir]\n";
        return 1;
    }

    std::string inputDir = argv[1];
    float sigma = std::atof(argv[2]);
    std::string outputDir = (argc >= 4) ? argv[3] : "../data/output";

    if (!fs::exists(outputDir)) fs::create_directories(outputDir);
    
    std::vector<std::string> files = getBMPFiles(inputDir);
    for (const auto& file : files) {
        processBMP(file, outputDir, sigma);
    }
    
    return 0;
}