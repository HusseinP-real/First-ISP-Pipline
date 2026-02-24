/**
 * RGB到YCbCr颜色空间转换测试程序
 * 
 * 批量处理BMP图像文件，将RGB转换为YCbCr颜色空间
 * - 自动扫描指定目录下的所有BMP文件
 * - 对每个文件执行RGB到YCbCr转换
 * - 保存Y、Cb、Cr三个通道为PNG图像
 * - 输出统计信息和汇总报告
 */

#include "../core/bmp/converter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>

/**
 * 辅助函数：打印统计信息
 */
void printStatistics(uint8_t** channel, int width, int height, const std::string& name) {
    double sum = 0.0;
    double sum_sq = 0.0;
    uint8_t min_val = 255;
    uint8_t max_val = 0;
    int count = width * height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t val = channel[y][x];
            sum += val;
            sum_sq += val * val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    std::cout << "\n[" << name << " 通道统计]" << std::endl;
    std::cout << "  最小值: " << static_cast<int>(min_val) << std::endl;
    std::cout << "  最大值: " << static_cast<int>(max_val) << std::endl;
    std::cout << "  平均值: " << std::fixed << std::setprecision(2) << mean << std::endl;
    std::cout << "  标准差: " << std::fixed << std::setprecision(2) << std_dev << std::endl;
}

/**
 * 从BMP图像文件读取并转换
 * @param bmpPath BMP文件路径
 * @param verbose 是否输出详细信息（默认true）
 */
void testBMPImage(const std::string& bmpPath, bool verbose = true) {
    if (verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "测试2: BMP图像文件转换测试" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    // 使用OpenCV读取BMP文件
    cv::Mat bmp_image = cv::imread(bmpPath, cv::IMREAD_COLOR);
    
    if (bmp_image.empty()) {
        std::cerr << "错误: 无法读取BMP文件: " << bmpPath << std::endl;
        return;
    }
    
    int width = bmp_image.cols;
    int height = bmp_image.rows;
    
    if (verbose) {
        std::cout << "图像尺寸: " << width << " x " << height << std::endl;
        std::cout << "图像类型: " << bmp_image.type() << " (CV_8UC3)" << std::endl;
    }
    
    // 分配RGB内存
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
    
    // 将OpenCV的BGR图像转换为RGB（OpenCV使用BGR格式）
    for (int y = 0; y < height; y++) {
        const cv::Vec3b* row = bmp_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; x++) {
            rgb[y][x].B = row[x][0];  // OpenCV的B
            rgb[y][x].G = row[x][1];  // OpenCV的G
            rgb[y][x].R = row[x][2];  // OpenCV的R
        }
    }
    
    if (verbose) {
        std::cout << "RGB数据已加载" << std::endl;
        std::cout << "执行RGB到YCbCr转换..." << std::endl;
    }
    
    // 执行转换
    RGB_to_YCbCr(rgb, Y, Cb, Cr, width, height);
    
    if (verbose) {
        std::cout << "转换完成" << std::endl;
        // 打印统计信息
        printStatistics(Y, width, height, "Y");
        printStatistics(Cb, width, height, "Cb");
        printStatistics(Cr, width, height, "Cr");
    }
    
    // 创建OpenCV Mat用于保存
    cv::Mat Y_mat(height, width, CV_8UC1);
    cv::Mat Cb_mat(height, width, CV_8UC1);
    cv::Mat Cr_mat(height, width, CV_8UC1);
    
    for (int y = 0; y < height; y++) {
        uint8_t* Y_row = Y_mat.ptr<uint8_t>(y);
        uint8_t* Cb_row = Cb_mat.ptr<uint8_t>(y);
        uint8_t* Cr_row = Cr_mat.ptr<uint8_t>(y);
        
        for (int x = 0; x < width; x++) {
            Y_row[x] = Y[y][x];
            Cb_row[x] = Cb[y][x];
            Cr_row[x] = Cr[y][x];
        }
    }
    
    // 保存通道图像
    std::string output_dir = "data/output/";
    std::string base_name = bmpPath.substr(bmpPath.find_last_of("/\\") + 1);
    base_name = base_name.substr(0, base_name.find_last_of("."));
    
    cv::imwrite(output_dir + base_name + "_Y.png", Y_mat);
    cv::imwrite(output_dir + base_name + "_Cb.png", Cb_mat);
    cv::imwrite(output_dir + base_name + "_Cr.png", Cr_mat);
    
    // 计算统计信息用于验证
    double Y_mean = cv::mean(Y_mat)[0];
    double Cb_mean = cv::mean(Cb_mat)[0];
    double Cr_mean = cv::mean(Cr_mat)[0];
    
    if (verbose) {
        std::cout << "\n通道图像已保存到:" << std::endl;
        std::cout << "  Y通道:  " << output_dir + base_name + "_Y.png" << std::endl;
        std::cout << "  Cb通道: " << output_dir + base_name + "_Cb.png" << std::endl;
        std::cout << "  Cr通道: " << output_dir + base_name + "_Cr.png" << std::endl;
        
        // 验证检查
        std::cout << "\n[验证检查]" << std::endl;
        std::cout << "  Y通道平均值: " << std::fixed << std::setprecision(2) << Y_mean << " (应该在0-255范围内)" << std::endl;
        std::cout << "  Cb通道平均值: " << std::fixed << std::setprecision(2) << Cb_mean << " (应该接近128)" << std::endl;
        std::cout << "  Cr通道平均值: " << std::fixed << std::setprecision(2) << Cr_mean << " (应该接近128)" << std::endl;
        
        // 检查Cb/Cr是否在合理范围内
        if (std::abs(Cb_mean - 128.0) < 20.0 && std::abs(Cr_mean - 128.0) < 20.0) {
            std::cout << "  ✓ Cb/Cr平均值检查通过" << std::endl;
        } else {
            std::cout << "  ⚠ Cb/Cr平均值可能异常（检查图像内容）" << std::endl;
        }
    } else {
        // 简洁输出
        std::cout << "  已保存: " << base_name << "_Y.png, " << base_name << "_Cb.png, " << base_name << "_Cr.png" << std::endl;
        std::cout << "  统计: Y=" << std::fixed << std::setprecision(1) << Y_mean 
                  << ", Cb=" << Cb_mean << ", Cr=" << Cr_mean;
        if (std::abs(Cb_mean - 128.0) < 20.0 && std::abs(Cr_mean - 128.0) < 20.0) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ⚠" << std::endl;
        }
    }
    
    // 释放内存
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

/**
 * 获取目录下所有BMP文件
 */
std::vector<std::string> getBMPFiles(const std::string& directory) {
    std::vector<std::string> bmpFiles;
    DIR* dir = opendir(directory.c_str());
    
    if (dir == nullptr) {
        std::cerr << "错误: 无法打开目录: " << directory << std::endl;
        return bmpFiles;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // 检查是否是.bmp文件
        if (filename.length() > 4) {
            std::string ext = filename.substr(filename.length() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".bmp") {
                std::string fullPath = directory + "/" + filename;
                bmpFiles.push_back(fullPath);
            }
        }
    }
    
    closedir(dir);
    
    // 排序文件名
    std::sort(bmpFiles.begin(), bmpFiles.end());
    
    return bmpFiles;
}

/**
 * 批量处理BMP文件
 */
void testBatchBMPFiles(const std::string& inputDir) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "批量BMP文件转换测试" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 获取所有BMP文件
    std::vector<std::string> bmpFiles = getBMPFiles(inputDir);
    
    if (bmpFiles.empty()) {
        std::cerr << "错误: 在目录 " << inputDir << " 中未找到BMP文件" << std::endl;
        return;
    }
    
    std::cout << "找到 " << bmpFiles.size() << " 个BMP文件:" << std::endl;
    for (size_t i = 0; i < bmpFiles.size(); i++) {
        std::string filename = bmpFiles[i].substr(bmpFiles[i].find_last_of("/\\") + 1);
        std::cout << "  " << (i + 1) << ". " << filename << std::endl;
    }
    std::cout << std::endl;
    
    // 统计信息
    int successCount = 0;
    int failCount = 0;
    std::vector<std::string> failedFiles;
    
    // 处理每个文件
    for (size_t i = 0; i < bmpFiles.size(); i++) {
        std::string filename = bmpFiles[i].substr(bmpFiles[i].find_last_of("/\\") + 1);
        std::cout << "\n[" << (i + 1) << "/" << bmpFiles.size() << "] 处理文件: " << filename << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        try {
            testBMPImage(bmpFiles[i], false);  // 批量处理时使用简洁输出
            successCount++;
        } catch (const std::exception& e) {
            failCount++;
            failedFiles.push_back(filename);
            std::cerr << "✗ 文件处理失败: " << e.what() << std::endl;
        }
    }
    
    // 打印汇总报告
    std::cout << "\n========================================" << std::endl;
    std::cout << "批量处理汇总报告" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "总文件数: " << bmpFiles.size() << std::endl;
    std::cout << "成功: " << successCount << std::endl;
    std::cout << "失败: " << failCount << std::endl;
    
    if (!failedFiles.empty()) {
        std::cout << "\n失败的文件:" << std::endl;
        for (const auto& file : failedFiles) {
            std::cout << "  - " << file << std::endl;
        }
    }
    
    std::cout << "\n所有输出图像已保存到: data/output/" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "RGB到YCbCr颜色空间转换测试程序" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // BMP图像文件批量处理
    if (argc > 1) {
        // 如果提供了参数，检查是文件还是目录
        std::string path = argv[1];
        struct stat info;
        
        if (stat(path.c_str(), &info) != 0) {
            std::cerr << "错误: 无法访问路径: " << path << std::endl;
            return -1;
        }
        
        if (S_ISDIR(info.st_mode)) {
            // 是目录，批量处理
            testBatchBMPFiles(path);
        } else {
            // 是文件，处理单个文件
            testBMPImage(path);
        }
    } else {
        // 默认：批量处理data/input目录下的所有BMP文件
        std::string inputDir = "data/input";
        std::cout << "\n未指定文件/目录，批量处理目录: " << inputDir << std::endl;
        testBatchBMPFiles(inputDir);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "所有测试完成！" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
