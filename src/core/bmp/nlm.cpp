#include "nlm.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>

/*
 * ============================================================================
 * 积分图加速 Non-Local Means (NLM) 去噪算法
 * ============================================================================
 * 
 * 核心优化思想:
 * 原始 NLM 复杂度: O(N × S² × T²)，其中 N=像素数, S=搜索窗口, T=模板窗口
 * 积分图加速后: O(N × S²)，与模板大小 T 无关！
 * 
 * 算法原理:
 * 1. 对于每个位移向量 (dx, dy)，计算全图的差分平方图: D = (I - I_shifted)²
 * 2. 对 D 做积分图 (Summed Area Table)
 * 3. 用积分图在 O(1) 时间内计算任意矩形区域的和，即 Patch Distance
 * 4. 根据距离计算权重并累加
 * 
 * 这样，模板块大小 T 不再影响计算时间！
 * ============================================================================
 */

/**
 * @brief 计算图像的积分图 (Summed Area Table)
 * 
 * 积分图 SAT[y][x] = 原图中 (0,0) 到 (x,y) 矩形区域的像素和
 * 
 * 任意矩形 (x1,y1) 到 (x2,y2) 的和可以在 O(1) 时间内计算:
 * sum = SAT[y2][x2] - SAT[y1-1][x2] - SAT[y2][x1-1] + SAT[y1-1][x1-1]
 */
static cv::Mat computeIntegralImage(const cv::Mat& src) {
    cv::Mat integral;
    // OpenCV 的 integral 函数非常快（SIMD优化）
    cv::integral(src, integral, CV_64F);
    return integral;
}

/**
 * @brief 使用积分图在 O(1) 时间内计算矩形区域的和
 * 
 * @param integral 积分图 (比原图大1行1列)
 * @param x, y 矩形左上角坐标（原图坐标系）
 * @param width, height 矩形的宽和高
 * @return 矩形区域内的像素值之和
 */
static inline double getBoxSum(const cv::Mat& integral, int x, int y, int width, int height) {
    // 积分图比原图大 1 行 1 列，坐标需要偏移
    int x1 = x;
    int y1 = y;
    int x2 = x + width;
    int y2 = y + height;
    
    // 使用积分图公式计算矩形和
    double a = integral.at<double>(y1, x1);
    double b = integral.at<double>(y1, x2);
    double c = integral.at<double>(y2, x1);
    double d = integral.at<double>(y2, x2);
    
    return d - b - c + a;
}

cv::Mat denoiseLuminanceNLM(const cv::Mat& y_float, float h_strength, int templateWindow, int searchWindow) {
    // ========================================
    // 1. 输入验证
    // ========================================
    if (y_float.empty()) {
        std::cerr << "Error: Input image is empty for denoiseLuminanceNLM." << std::endl;
        return cv::Mat();
    }
    if (y_float.type() != CV_32F) {
        std::cerr << "Error: Input image type is not CV_32F." << std::endl;
        return cv::Mat();
    }

    // ========================================
    // 2. 参数准备
    // ========================================
    int rows = y_float.rows;
    int cols = y_float.cols;
    
    int halfTemplate = templateWindow / 2;
    int halfSearch = searchWindow / 2;

    // h 参数处理
    float h = h_strength / 255.0f;
    float h2 = h * h;
    float inv_h2 = 1.0f / h2;
    
    // 模板块大小（用于归一化距离）
    float patchSize = static_cast<float>(templateWindow * templateWindow);

    // ========================================
    // 3. 边界填充
    // ========================================
    int borderSize = halfSearch + halfTemplate;
    cv::Mat padded;
    cv::copyMakeBorder(y_float, padded, borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT101);

    int paddedRows = padded.rows;
    int paddedCols = padded.cols;

    // 创建输出图像和累加器
    cv::Mat weightSum = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat pixelSum = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat maxWeight = cv::Mat::zeros(rows, cols, CV_64F);

    // 显示信息
    int num_threads = omp_get_max_threads();
    std::cout << "NLM (Integral Image Acceleration): Using " << num_threads << " threads" << std::endl;
    std::cout << "Image size: " << cols << "x" << rows << std::endl;
    std::cout << "Parameters: h=" << h_strength << ", template=" << templateWindow << ", search=" << searchWindow << std::endl;
    
    int totalShifts = (2 * halfSearch + 1) * (2 * halfSearch + 1) - 1;
    int processedShifts = 0;
    int lastProgress = 0;

    // ========================================
    // 4. 积分图加速的 NLM 主算法
    // ========================================
    // 遍历所有位移向量 (dx, dy)
    for (int dy = -halfSearch; dy <= halfSearch; ++dy) {
        for (int dx = -halfSearch; dx <= halfSearch; ++dx) {
            // 跳过零位移（中心像素后面单独处理）
            if (dy == 0 && dx == 0) continue;

            // Step 1: 计算差分平方图
            // 对于位移 (dx, dy)，计算 D(x,y) = (padded(x,y) - padded(x+dx, y+dy))²
            cv::Mat diffSquared = cv::Mat::zeros(paddedRows, paddedCols, CV_32F);
            
            // 有效区域（确保两个块都在 padded 范围内）
            int startY = std::max(0, -dy);
            int endY = std::min(paddedRows, paddedRows - dy);
            int startX = std::max(0, -dx);
            int endX = std::min(paddedCols, paddedCols - dx);

            #pragma omp parallel for schedule(static)
            for (int y = startY; y < endY; ++y) {
                const float* row1 = padded.ptr<float>(y);
                const float* row2 = padded.ptr<float>(y + dy);
                float* rowDiff = diffSquared.ptr<float>(y);
                
                for (int x = startX; x < endX; ++x) {
                    float diff = row1[x] - row2[x + dx];
                    rowDiff[x] = diff * diff;
                }
            }

            // Step 2: 计算差分平方图的积分图
            cv::Mat integralDiff = computeIntegralImage(diffSquared);

            // Step 3: 对每个像素，使用积分图计算 patch distance
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < rows; ++y) {
                double* pWeightSum = weightSum.ptr<double>(y);
                double* pPixelSum = pixelSum.ptr<double>(y);
                double* pMaxWeight = maxWeight.ptr<double>(y);

                for (int x = 0; x < cols; ++x) {
                    // 当前像素在 padded 中的坐标
                    int padY = y + borderSize;
                    int padX = x + borderSize;

                    // 邻居像素在 padded 中的坐标
                    int neighborY = padY + dy;
                    int neighborX = padX + dx;

                    // 计算以 (padY, padX) 为中心的 templateWindow × templateWindow 区域的差分平方和
                    // 区域左上角坐标
                    int boxY = padY - halfTemplate;
                    int boxX = padX - halfTemplate;

                    // 使用积分图在 O(1) 时间内计算 patch distance
                    double dist = getBoxSum(integralDiff, boxX, boxY, templateWindow, templateWindow);
                    
                    // 归一化距离
                    dist /= patchSize;

                    // 计算权重
                    double weight = std::exp(-dist * inv_h2);

                    // 获取邻居像素值
                    float neighborVal = padded.at<float>(neighborY, neighborX);

                    // 累加
                    pWeightSum[x] += weight;
                    pPixelSum[x] += weight * neighborVal;

                    if (weight > pMaxWeight[x]) {
                        pMaxWeight[x] = weight;
                    }
                }
            }

            processedShifts++;
            int progress = (100 * processedShifts) / totalShifts;
            if (progress >= lastProgress + 10) {
                std::cout << "NLM Denoising progress: " << progress << "%" << std::endl;
                lastProgress = progress;
            }
        }
    }

    // ========================================
    // 5. 处理中心像素并归一化
    // ========================================
    cv::Mat denoised = cv::Mat::zeros(rows, cols, CV_32F);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < rows; ++y) {
        float* pDenoised = denoised.ptr<float>(y);
        double* pWeightSum = weightSum.ptr<double>(y);
        double* pPixelSum = pixelSum.ptr<double>(y);
        double* pMaxWeight = maxWeight.ptr<double>(y);

        for (int x = 0; x < cols; ++x) {
            int padY = y + borderSize;
            int padX = x + borderSize;
            float centerVal = padded.at<float>(padY, padX);

            // 中心像素使用最大权重
            pWeightSum[x] += pMaxWeight[x];
            pPixelSum[x] += pMaxWeight[x] * centerVal;

            // 归一化
            if (pWeightSum[x] > 1e-10) {
                pDenoised[x] = static_cast<float>(pPixelSum[x] / pWeightSum[x]);
            } else {
                pDenoised[x] = centerVal;
            }
        }
    }

    std::cout << "NLM Denoising complete!" << std::endl;
    return denoised;
}

/*
 * ============================================================================
 * 优化版 引导滤波 (Guided Filter)
 * ============================================================================
 * 使用 CV_32F 而非 CV_64F，速度更快
 * ============================================================================
 */

cv::Mat denoiseChromaGuided(const cv::Mat& guide_Y, const cv::Mat& src_chroma, int radius, float eps) {
    // ========================================
    // 1. 输入验证
    // ========================================
    if (guide_Y.empty()) {
        std::cerr << "Error: Guide image (Y channel) is empty for denoiseChromaGuided." << std::endl;
        return cv::Mat();
    }
    if (src_chroma.empty()) {
        std::cerr << "Error: Source chroma channel is empty for denoiseChromaGuided." << std::endl;
        return cv::Mat();
    }
    if (guide_Y.type() != CV_32F) {
        std::cerr << "Error: Guide image type is not CV_32F. Got type: " << guide_Y.type() << std::endl;
        return cv::Mat();
    }
    if (src_chroma.type() != CV_32F) {
        std::cerr << "Error: Source chroma type is not CV_32F. Got type: " << src_chroma.type() << std::endl;
        return cv::Mat();
    }
    if (guide_Y.size() != src_chroma.size()) {
        std::cerr << "Error: Guide and source chroma dimensions do not match." << std::endl;
        return cv::Mat();
    }

    // ========================================
    // 2. 准备工作
    // ========================================
    float eps_squared = eps * eps;
    cv::Size ksize(2 * radius + 1, 2 * radius + 1);
    
    const cv::Mat& I = guide_Y;
    const cv::Mat& p = src_chroma;

    // ========================================
    // 3. 计算局部统计量
    // ========================================
    cv::Mat mean_I, mean_p, mean_II, mean_Ip;
    
    cv::boxFilter(I, mean_I, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(p, mean_p, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    
    cv::Mat II, Ip;
    cv::multiply(I, I, II);
    cv::multiply(I, p, Ip);
    
    cv::boxFilter(II, mean_II, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(Ip, mean_Ip, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

    // ========================================
    // 4. 计算协方差和方差
    // ========================================
    cv::Mat var_I, cov_Ip;
    
    cv::multiply(mean_I, mean_I, var_I);
    cv::subtract(mean_II, var_I, var_I);
    
    cv::multiply(mean_I, mean_p, cov_Ip);
    cv::subtract(mean_Ip, cov_Ip, cov_Ip);

    // ========================================
    // 5. 计算线性系数 a 和 b
    // ========================================
    cv::Mat a, b;
    
    cv::add(var_I, cv::Scalar(eps_squared), var_I);
    cv::divide(cov_Ip, var_I, a);
    
    cv::multiply(a, mean_I, b);
    cv::subtract(mean_p, b, b);

    // ========================================
    // 6. 对 a 和 b 进行平均
    // ========================================
    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(b, mean_b, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

    // ========================================
    // 7. 计算最终输出
    // ========================================
    cv::Mat q;
    cv::multiply(mean_a, I, q);
    cv::add(q, mean_b, q);

    return q;
}

/*
 * ============================================================================
 * LUT 优化的双边滤波 (Bilateral Filter)
 * ============================================================================
 * 
 * 双边滤波原理:
 * 对于每个像素，输出值是其邻域像素的加权平均，权重由两部分组成:
 * 1. 空间权重 (Spatial Weight): 基于像素间的空间距离
 * 2. 值域权重 (Range Weight): 基于像素值的差异
 * 
 * 公式: q(i) = (1/W) * Σ G_s(||i-j||) * G_r(|I(i)-I(j)|) * I(j)
 * 其中 W = Σ G_s(||i-j||) * G_r(|I(i)-I(j)|)
 * 
 * LUT 优化:
 * - 空间权重只与位置有关，可预先计算整个窗口的权重表
 * - 值域权重可以量化到 LUT 中，避免重复计算 exp()
 * ============================================================================
 */

cv::Mat myBilateralFilter(const cv::Mat& src, int radius, float sigma_spatial, float sigma_range) {
    // ========================================
    // 1. 输入验证
    // ========================================
    if (src.empty()) {
        std::cerr << "Error: Input image is empty for myBilateralFilter." << std::endl;
        return cv::Mat();
    }
    if (src.type() != CV_32F) {
        std::cerr << "Error: Input image type is not CV_32F for myBilateralFilter." << std::endl;
        return cv::Mat();
    }

    int rows = src.rows;
    int cols = src.cols;
    int windowSize = 2 * radius + 1;

    // ========================================
    // 2. 预计算空间权重 LUT (Spatial Gaussian LUT)
    // ========================================
    // 空间权重只与像素位置有关，可以预先计算整个窗口
    std::vector<float> spatialLUT(windowSize * windowSize);
    float spatialCoeff = -0.5f / (sigma_spatial * sigma_spatial);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float distSq = static_cast<float>(dx * dx + dy * dy);
            int idx = (dy + radius) * windowSize + (dx + radius);
            spatialLUT[idx] = std::exp(spatialCoeff * distSq);
        }
    }

    // ========================================
    // 3. 预计算值域权重 LUT (Range Gaussian LUT)
    // ========================================
    // 值域差异在 [0, 1] 范围内，量化到 256 级
    const int RANGE_LUT_SIZE = 256;
    std::vector<float> rangeLUT(RANGE_LUT_SIZE);
    float rangeCoeff = -0.5f / (sigma_range * sigma_range);
    
    for (int i = 0; i < RANGE_LUT_SIZE; ++i) {
        // 将索引映射到 [0, 1] 范围的差值
        float diff = static_cast<float>(i) / (RANGE_LUT_SIZE - 1);
        rangeLUT[i] = std::exp(rangeCoeff * diff * diff);
    }

    // ========================================
    // 4. 边界填充
    // ========================================
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, radius, radius, radius, radius, cv::BORDER_REFLECT101);

    // ========================================
    // 5. 双边滤波主循环
    // ========================================
    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_32F);

    int num_threads = omp_get_max_threads();
    std::cout << "Bilateral Filter (LUT Optimized): Using " << num_threads << " threads" << std::endl;
    std::cout << "Image size: " << cols << "x" << rows << std::endl;
    std::cout << "Parameters: radius=" << radius << ", sigma_spatial=" << sigma_spatial 
              << ", sigma_range=" << sigma_range << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rows; ++y) {
        float* pDst = dst.ptr<float>(y);
        
        for (int x = 0; x < cols; ++x) {
            // 当前像素在 padded 图像中的坐标
            int padY = y + radius;
            int padX = x + radius;
            
            // 中心像素值
            float centerVal = padded.at<float>(padY, padX);
            
            float weightSum = 0.0f;
            float valueSum = 0.0f;
            
            // 遍历邻域窗口
            for (int dy = -radius; dy <= radius; ++dy) {
                const float* pRow = padded.ptr<float>(padY + dy);
                int lutRowOffset = (dy + radius) * windowSize;
                
                for (int dx = -radius; dx <= radius; ++dx) {
                    float neighborVal = pRow[padX + dx];
                    
                    // 空间权重 (从 LUT 获取)
                    float spatialWeight = spatialLUT[lutRowOffset + (dx + radius)];
                    
                    // 值域权重 (从 LUT 获取)
                    float diff = std::abs(centerVal - neighborVal);
                    // 将差值映射到 LUT 索引 [0, 255]
                    int rangeIdx = static_cast<int>(diff * (RANGE_LUT_SIZE - 1));
                    rangeIdx = std::min(rangeIdx, RANGE_LUT_SIZE - 1);
                    float rangeWeight = rangeLUT[rangeIdx];
                    
                    // 组合权重
                    float weight = spatialWeight * rangeWeight;
                    
                    weightSum += weight;
                    valueSum += weight * neighborVal;
                }
            }
            
            // 归一化输出
            if (weightSum > 1e-10f) {
                pDst[x] = valueSum / weightSum;
            } else {
                pDst[x] = centerVal;
            }
        }
    }

    std::cout << "Bilateral Filter complete!" << std::endl;
    return dst;
}