#include "../core/raw_reader.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取RAW图像 (512x500, 16bit, 28帧)
    std::string inputFile = "data/input/raw1.raw";
    int frameIndex = 0;  // 读取第一帧（索引从0开始）
    
    cv::Mat rawImage = readRawToMat(inputFile, 512, 500, frameIndex);
    
    if (rawImage.empty()) {
        std::cerr << "Failed to read RAW image: " << inputFile << std::endl;
        return -1;
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Successfully read RAW image: " << inputFile << std::endl;
    std::cout << "Frame index: " << frameIndex << " (file contains 28 frames)" << std::endl;
    std::cout << "Image size: " << rawImage.cols << "x" << rawImage.rows << std::endl;

    // ... 读取代码后 ...

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(rawImage, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "=== Data Inspection ===" << std::endl;
    std::cout << "Min Value: " << minVal << std::endl;
    std::cout << "Max Value: " << maxVal << std::endl;

    // 判读标准：
    // 如果 Max Value 是 0 -> 读取失败，全黑。
    // 如果 Max Value 很大 (例如 > 60000) -> 可能是乱码或字节序错误。
    // 如果 Max Value 是 1023, 4095, 16383 附近 -> 恭喜！读取完美成功，分别是 10位/12位/14位数据。
    
    // 归一化以便保存（16位数据需要归一化到0-255范围）
    // 如果读取正确，保存的图像应该是黑白马赛克图（有很多噪点感觉的图）
    cv::Mat normalizedImage;
    rawImage.convertTo(normalizedImage, CV_8U, 255.0 / 65535.0);
    
    // 保存归一化后的图像到输出目录
    std::string outputFile = "data/output/raw1_normalized.png";
    bool success = cv::imwrite(outputFile, normalizedImage);
    
    if (success) {
        std::cout << "Successfully saved normalized image to: " << outputFile << std::endl;
        std::cout << "If the image shows a black and white mosaic pattern (with noise), the reading is correct!" << std::endl;
    } else {
        std::cerr << "Failed to save image to: " << outputFile << std::endl;
        return -1;
    }
    
    return 0;
}