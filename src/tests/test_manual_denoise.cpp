/**
 * Manual DCT & DWT Denoising Test Program
 *
 * Workflow:
 * 1. Batch process BMP files from data/input.
 * 2. Convert RGB to YCbCr using custom converter.h.
 * 3. Apply Manual Sliding Window DCT Denoising on Y channel.
 * 4. Apply Manual DWT (BayesShrink) Denoising on Y channel.
 * 5. Save comparison results.
 */

#include "../core/bmp/converter.h"
#include "../core/bmp/dct.h"
#include "../core/bmp/nlm.h"

#include <opencv2/opencv.hpp> // Only for IO
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// Helper to get all BMP files
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

void processBMP(const std::string& bmpPath, const std::string& outputDir) {
    std::string filename = bmpPath.substr(bmpPath.find_last_of("/\\") + 1);
    std::string baseName = filename.substr(0, filename.find_last_of("."));
    
    std::cout << "\nprocessing: " << filename << std::endl;

    // Load Image
    cv::Mat bmp_image = cv::imread(bmpPath, cv::IMREAD_COLOR);
    if (bmp_image.empty()) {
        std::cerr << "Error: Could not read image: " << bmpPath << std::endl;
        return;
    }
    int width = bmp_image.cols;
    int height = bmp_image.rows;

    // Prepare Memory
    RGBPixel** rgb = new RGBPixel*[height];
    uint8_t** Y = new uint8_t*[height];
    uint8_t** Cb = new uint8_t*[height];
    uint8_t** Cr = new uint8_t*[height];
    for (int i = 0; i < height; i++) {
        rgb[i] = new RGBPixel[width];
        Y[i] = new uint8_t[width];
        Cb[i] = new uint8_t[width];
        Cr[i] = new uint8_t[width];
    }

    // BGR -> RGB & Fill
    for (int y = 0; y < height; y++) {
        const cv::Vec3b* row = bmp_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; x++) {
            rgb[y][x].B = row[x][0];
            rgb[y][x].G = row[x][1];
            rgb[y][x].R = row[x][2];
        }
    }

    // RGB -> YCbCr
    RGB_to_YCbCr(rgb, Y, Cb, Cr, width, height);

    // Convert Y to float vector for processing
    std::vector<float> y_data(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            y_data[y * width + x] = Y[y][x] / 255.0f; 
        }
    }

    // --- DCT Denoising ---
    std::cout << "  > Running DCT Denoising..." << std::endl;
    auto start_dct = std::chrono::high_resolution_clock::now();
    std::vector<float> y_dct(width * height);
    float sigma = 10.0f / 255.0f; // Assumed noise level
    denoiseDCT(y_data, y_dct, width, height, sigma);
    auto end_dct = std::chrono::high_resolution_clock::now();
    std::cout << "    DCT Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dct - start_dct).count() << " ms" << std::endl;




    // --- Chroma Denoising (Guided Filter) ---
    std::cout << "  > Running Chroma Denoising (Guided Filter)..." << std::endl;
    cv::Mat y_denoised_float(height, width, CV_32F);
    cv::Mat cb_float(height, width, CV_32F);
    cv::Mat cr_float(height, width, CV_32F);

    // Fill Mats
    for (int y = 0; y < height; ++y) {
        float* pY = y_denoised_float.ptr<float>(y);
        float* pCb = cb_float.ptr<float>(y);
        float* pCr = cr_float.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            pY[x] = std::clamp(y_dct[y * width + x], 0.0f, 1.0f);
            pCb[x] = Cb[y][x] / 255.0f;
            pCr[x] = Cr[y][x] / 255.0f;
        }
    }

    // Apply Guided Filter
    int guided_radius = 8;
    float guided_eps = 0.02f;
    cv::Mat cb_denoised = denoiseChromaGuided(y_denoised_float, cb_float, guided_radius, guided_eps);
    cv::Mat cr_denoised = denoiseChromaGuided(y_denoised_float, cr_float, guided_radius, guided_eps);

    // --- Reconstruct Full Color Image ---
    // Update Cb/Cr buffers with denoised values
    // Update Y buffer with DCT denoised values
    for (int y = 0; y < height; ++y) {
        const float* pCb = cb_denoised.ptr<float>(y);
        const float* pCr = cr_denoised.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            Y[y][x] = static_cast<uint8_t>(y_dct[y * width + x] * 255.0f);
            Cb[y][x] = static_cast<uint8_t>(std::clamp(pCb[x] * 255.0f, 0.0f, 255.0f));
            Cr[y][x] = static_cast<uint8_t>(std::clamp(pCr[x] * 255.0f, 0.0f, 255.0f));
        }
    }

    // YCbCr -> RGB
    YCbCr_to_RGB(Y, Cb, Cr, rgb, width, height);

    // Save outputs
    // 1. Y Channel (DCT)
    cv::Mat dct_img(height, width, CV_8UC1);
    for (int i = 0; i < height * width; ++i) {
        dct_img.data[i] = static_cast<uint8_t>(std::clamp(y_dct[i] * 255.0f, 0.0f, 255.0f));
    }
    cv::imwrite(outputDir + "/" + baseName + "_Y_dct.png", dct_img);

    // 2. Full Color (DCT + Guided)
    cv::Mat output_image(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        cv::Vec3b* row = output_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            row[x][0] = rgb[y][x].B;
            row[x][1] = rgb[y][x].G;
            row[x][2] = rgb[y][x].R;
        }
    }
    cv::imwrite(outputDir + "/" + baseName + "_dct_full.png", output_image);

    std::cout << "  Saved results to " << outputDir << std::endl;

    // Cleanup
    for (int i = 0; i < height; i++) {
        delete[] rgb[i];
        delete[] Y[i];
        delete[] Cb[i];
        delete[] Cr[i];
    }
    delete[] rgb;
    delete[] Y;
    delete[] Cb;
    delete[] Cr;
}

int main(int argc, char* argv[]) {
    std::string inputDir = "data/input";
    std::string outputDir = "data/output";

    if (argc > 1) inputDir = argv[1];
    if (argc > 2) outputDir = argv[2];

    std::cout << "Manual DCT & DWT Denoising Test" << std::endl;
    std::cout << "Input: " << inputDir << std::endl;
    std::cout << "Output: " << outputDir << std::endl;

    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    std::vector<std::string> files = getBMPFiles(inputDir);
    for (const auto& file : files) {
        processBMP(file, outputDir);
    }

    return 0;
}
