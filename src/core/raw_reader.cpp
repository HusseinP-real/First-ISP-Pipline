#include "raw_reader.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <iostream>

cv::Mat readRawToMat(const std::string& filename, int width, int height, int frameIndex) {
    // open file in binary
    std::ifstream file(filename, std::ios::binary);

    // check if is opened
    if (!file.is_open()) {
        std::cout << "Error: Failed to open file: " << filename << std::endl;
        return cv::Mat();
    }

    // calculate frame size (16bit = 2 bytes per pixel)
    size_t frameSize = static_cast<size_t>(width) * height * 2;
    
    // seek to the start of the requested frame (28 frames total)
    file.seekg(frameIndex * frameSize, std::ios::beg);

    // create a raw image matrix
    cv::Mat rawImage(height, width, CV_16U);

    // read one frame of raw image data
    file.read(reinterpret_cast<char*>(rawImage.data), frameSize);

    file.close();
    return rawImage;
}