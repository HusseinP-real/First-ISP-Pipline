/**
 * BM3D (Y) + Guided Filter (Chroma) Denoising Test Program
 *
 * Workflow:
 * 1. Batch process BMP files from data/input.
 * 2. Convert RGB to YCbCr using custom converter.h (separates channels).
 * 3. Apply BM3D Denoising on Y (Luminance) channel.
 * 4. Apply Guided Filter Denoising on Cb/Cr (Chroma) channels using denoised Y as guide.
 * 5. Convert back to RGB and save results.
 */

#include "../core/bmp/converter.h"
#include "../core/bmp/nlm.h" // Still needed for guided filter helpers if they are there, or just general utils
#include "../../bm3d/include/bm3d_denoiser.hpp" // BM3D include
#include <opencv2/opencv.hpp>
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

/**
 * Helper to get all BMP files in a directory
 */
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

/**
 * Process a single BMP file with full YCbCr denoising pipeline
 */
void processBMP(const std::string& bmpPath, const std::string& outputDir) {
    std::string filename = bmpPath.substr(bmpPath.find_last_of("/\\") + 1);
    std::string baseName = filename.substr(0, filename.find_last_of("."));
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing: " << filename << std::endl;
    std::cout << "========================================" << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();

    // 1. Load Image using OpenCV
    cv::Mat bmp_image = cv::imread(bmpPath, cv::IMREAD_COLOR);
    if (bmp_image.empty()) {
        std::cerr << "Error: Could not read image: " << bmpPath << std::endl;
        return;
    }
    int width = bmp_image.cols;
    int height = bmp_image.rows;
    std::cout << "Image size: " << width << " x " << height << std::endl;

    // 2. Prepare Memory for RGB -> YCbCr Conversion
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

    // 3. Fill RGB data from OpenCV Mat (BGR -> RGB)
    for (int y = 0; y < height; y++) {
        const cv::Vec3b* row = bmp_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; x++) {
            rgb[y][x].B = row[x][0];
            rgb[y][x].G = row[x][1];
            rgb[y][x].R = row[x][2];
        }
    }

    // 4. Convert to YCbCr using custom converter
    RGB_to_YCbCr(rgb, Y, Cb, Cr, width, height);

    // 5. Convert all channels to OpenCV Mat (CV_8U)
    // BM3D implementation expects CV_8U or CV_64F. We'll use CV_8U initially.
    cv::Mat y_mat(height, width, CV_8UC1);
    cv::Mat cb_mat(height, width, CV_8UC1);
    cv::Mat cr_mat(height, width, CV_8UC1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            y_mat.at<uint8_t>(y, x) = Y[y][x];
            cb_mat.at<uint8_t>(y, x) = Cb[y][x];
            cr_mat.at<uint8_t>(y, x) = Cr[y][x];
        }
    }
    
    // For Guided Filter, we still use Float32 [0, 1] as per original implementation
    cv::Mat cb_float, cr_float;
    cb_mat.convertTo(cb_float, CV_32F, 1.0 / 255.0);
    cr_mat.convertTo(cr_float, CV_32F, 1.0 / 255.0);

    // ========================================
    // 6. Apply BM3D Denoising on Y channel
    // ========================================
    std::cout << "\n[Step 1] BM3D Denoising on Y channel..." << std::endl;
    auto start_bm3d = std::chrono::high_resolution_clock::now();
    
    float sigma = 25.0f; // Default Sigma for BM3D
    // Correction for Y channel noise: Y = 0.299R + 0.587G + 0.114B
    // sigma_y approx 0.67 * sigma
    float sigma_y = sigma * 0.67f; 
    
    BM3D_Denoiser denoiser;
    cv::Mat y_denoised_8u = denoiser.denoise(y_mat, sigma_y);
    
    // Convert Y denoised to float for Guided Filter guide
    cv::Mat y_denoised_float;
    y_denoised_8u.convertTo(y_denoised_float, CV_32F, 1.0 / 255.0);

    auto end_bm3d = std::chrono::high_resolution_clock::now();
    auto bm3d_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_bm3d - start_bm3d);
    std::cout << "BM3D completed in " << bm3d_duration.count() << " ms" << std::endl;

    // ========================================
    // 7. Apply Guided Filter on Cb and Cr channels
    // ========================================
    std::cout << "\n[Step 2] Guided Filter Denoising on Cb/Cr channels..." << std::endl;
    auto start_guided = std::chrono::high_resolution_clock::now();
    
    int guided_radius = 8;     // Guided filter radius
    float guided_eps = 0.02f;  // Regularization parameter
    
    // Use denoised Y as guide image
    cv::Mat cb_denoised_float = denoiseChromaGuided(y_denoised_float, cb_float, guided_radius, guided_eps);
    cv::Mat cr_denoised_float = denoiseChromaGuided(y_denoised_float, cr_float, guided_radius, guided_eps);
    
    auto end_guided = std::chrono::high_resolution_clock::now();
    auto guided_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_guided - start_guided);
    std::cout << "Guided Filter completed in " << guided_duration.count() << " ms" << std::endl;

    // ========================================
    // 8. Convert results back to 8-bit for saving
    // ========================================
    cv::Mat cb_denoised_8u, cr_denoised_8u;
    cb_denoised_float.convertTo(cb_denoised_8u, CV_8U, 255.0);
    cr_denoised_float.convertTo(cr_denoised_8u, CV_8U, 255.0);

    // ========================================
    // 9. Convert back to RGB and save full result
    // ========================================
    // Update denoised values back to arrays
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Y[y][x] = y_denoised_8u.at<uint8_t>(y, x);
            Cb[y][x] = cb_denoised_8u.at<uint8_t>(y, x);
            Cr[y][x] = cr_denoised_8u.at<uint8_t>(y, x);
        }
    }
    
    // Convert YCbCr back to RGB
    YCbCr_to_RGB(Y, Cb, Cr, rgb, width, height);
    
    // Create output image
    cv::Mat output_image(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        cv::Vec3b* row = output_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; x++) {
            row[x][0] = rgb[y][x].B;  // BGR order for OpenCV
            row[x][1] = rgb[y][x].G;
            row[x][2] = rgb[y][x].R;
        }
    }
    
    cv::imwrite(outputDir + "/" + baseName + "_bm3d_full_denoised.png", output_image);

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

    std::cout << "\nResults saved to " << outputDir << ":" << std::endl;
    std::cout << "  - " << baseName << "_bm3d_full_denoised.png (RGB)" << std::endl;
    std::cout << "\nTotal processing time: " << total_duration.count() << " ms" << std::endl;

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

    if (argc > 1) {
        inputDir = argv[1];
    }
    if (argc > 2) {
        outputDir = argv[2];
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  BM3D + Guided Filter Denoising Test" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Input directory:  " << inputDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << std::endl;
    std::cout << "Algorithms:" << std::endl;
    std::cout << "  - Y channel:    BM3D (local impl, sigma corrected)" << std::endl;
    std::cout << "  - Cb/Cr:        Guided Filter (manual impl)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Ensure output directory exists
    if (!fs::exists(outputDir)) {
        if (!fs::create_directories(outputDir)) {
             std::cerr << "Error: Could not create output directory: " << outputDir << std::endl;
             return -1;
        }
        std::cout << "Created output directory: " << outputDir << std::endl;
    }
    
    std::vector<std::string> files = getBMPFiles(inputDir);
    if (files.empty()) {
        std::cerr << "No BMP files found in " << inputDir << std::endl;
        return 0;
    }

    std::cout << "Found " << files.size() << " BMP file(s). Starting processing..." << std::endl;

    for (const auto& file : files) {
        processBMP(file, outputDir);
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "All processing complete!" << std::endl;
    std::cout << "============================================" << std::endl;
    return 0;
}
