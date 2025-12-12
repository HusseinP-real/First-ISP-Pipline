#include "../core/raw_reader.h"
#include "../core/blc.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <iomanip> // 用于格式化输出

int main() {
    // 1. 读取原始 RAW 图像 (只读一次，作为母版)
    std::string inputFile = "data/input/raw1.raw";
    int frameIndex = 0;
    
    cv::Mat originalRaw = readRawToMat(inputFile, 512, 500, frameIndex);
    
    if (originalRaw.empty()) {
        std::cerr << "Failed to read RAW image: " << inputFile << std::endl;
        return -1;
    }

    std::cout << "Successfully read RAW image. Size: " << originalRaw.cols << "x" << originalRaw.rows << std::endl;
    
    // 检查是否为 16位
    if (originalRaw.type() != CV_16UC1) {
        std::cerr << "Error: Expected 16-bit image." << std::endl;
        return -1;
    }

    // 定义要测试的三个黑电平值
    std::vector<int> test_values = {135, 140, 145};

    // 图像信息
    imageInfo info;
    info.width = originalRaw.cols;
    info.height = originalRaw.rows;
    info.ob_rows = 0;

    std::cout << "\n================ START TUNING ================" << std::endl;

    for (int val : test_values) {
        // ---------------------------------------------------------
        // 重要：每次测试前，必须从母版 Clone 一份新的数据！
        // 否则第二次循环会在第一次减过的基础上继续减，数据就错了。
        // ---------------------------------------------------------
        cv::Mat workImg = originalRaw.clone();

        // 构造 BLC 参数
        blackLevels bls;
        bls.r  = (float)val;
        bls.gr = (float)val;
        bls.gb = (float)val;
        bls.b  = (float)val;

        // 执行校正
        uint16_t* rawData = reinterpret_cast<uint16_t*>(workImg.data);
        applyBlc(rawData, info, bls);

        // --- 统计核心指标 (利用 OpenCV 函数加速) ---
        
        // 1. 计算均值 (Mean)
        cv::Scalar meanScalar = cv::mean(workImg);
        double meanVal = meanScalar[0];

        // 2. 计算零值比例 (Zero Ratio)
        // countNonZero 统计非0像素，总数减去它就是0像素个数
        int totalPixels = workImg.total();
        int zeroCount = totalPixels - cv::countNonZero(workImg);
        double zeroRatio = (double)zeroCount / totalPixels * 100.0;

        // 3. 获取 Min/Max (虽然用处不大，带着也可以)
        double minVal, maxVal;
        cv::minMaxLoc(workImg, &minVal, &maxVal);

        // --- 打印结果报告 ---
        std::cout << ">>> Testing Black Level: [" << val << "]" << std::endl;
        std::cout << "    Min: " << minVal << " | Max: " << maxVal << std::endl;
        std::cout << "    [Key] Mean Value: " << std::fixed << std::setprecision(4) << meanVal 
                  << " (Ideal: 0.5 ~ 2.0)" << std::endl;
        std::cout << "    [Key] Zero Ratio: " << std::fixed << std::setprecision(2) << zeroRatio 
                  << "% (Too high means details lost)" << std::endl;

        // --- 保存可视化图像 ---
        
        // 为了让人眼能看清暗部，我们需要加“数字增益 (Digital Gain)”
        // 原始数据太暗了，保存成PNG就是全黑。这里我们放大 64 倍来看看底噪。
        cv::Mat visualImg;
        workImg.convertTo(visualImg, -1, 64.0); // 乘以 64， -1 表示保持类型不变

        // 转为 8位 用于保存
        // 注意：这里保存的图是“提亮版”，仅用于肉眼观察底噪情况
        cv::Mat saveImg;
        visualImg.convertTo(saveImg, CV_8U, 255.0 / 65535.0);

        std::string outName = "data/output/blc_" + std::to_string(val) + "_gain64x.png";
        cv::imwrite(outName, saveImg);
        std::cout << "    Saved visual check image to: " << outName << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
    }

    return 0;
}