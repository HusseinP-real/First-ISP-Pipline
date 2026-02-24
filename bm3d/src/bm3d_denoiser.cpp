/**
 * @file bm3d_denoiser.cpp
 * @brief Implementation of the BM3D Image Denoising Algorithm
 * 
 * Optimized version with:
 * - Pre-computed cosine tables for DCT (fetched ONCE before hot loops)
 * - Pre-allocated buffers to avoid heap allocations in hot loops
 * - Optimized memory access patterns using ptr<double>()
 * - OpenMP parallelization with minimal lock contention
 */

#include "bm3d_denoiser.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// ======================== DCT Cosine Table (Singleton) ========================

DCTCosineTable& DCTCosineTable::instance() {
    static DCTCosineTable inst;
    return inst;
}

const std::vector<double>& DCTCosineTable::getCosTable(int n) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto it = m_tables.find(n);
    if (it != m_tables.end()) {
        return it->second;
    }
    
    // Compute cosine table: cos(π * k * (2*i + 1) / (2*n)) for all k, i in [0, n)
    // Store as flattened 2D array: table[k * n + i]
    std::vector<double> table(n * n);
    const double factor = M_PI / (2.0 * n);
    
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            table[k * n + i] = std::cos(factor * k * (2.0 * i + 1.0));
        }
    }
    
    m_tables[n] = std::move(table);
    return m_tables[n];
}

// ======================== Constructors ========================

BM3D_Denoiser::BM3D_Denoiser()
    : m_step1BlockSize(8)
    , m_step1WindowSize(39)
    , m_step1MaxMatch(16)
    , m_step1ThreDist(2500.0)
    , m_step1SpdupFactor(3)
    , m_step2BlockSize(8)
    , m_step2WindowSize(39)
    , m_step2MaxMatch(32)
    , m_step2ThreDist(400.0)
    , m_step2SpdupFactor(3)
    , m_kaiserBeta(2.0)
    , m_lambda2d(2.0)
    , m_lambda3d(2.7)
    , m_sigma(25.0)
{
    // Pre-warm cosine tables for common sizes (done once at construction)
    DCTCosineTable::instance().getCosTable(16);
    DCTCosineTable::instance().getCosTable(32);
}

BM3D_Denoiser::BM3D_Denoiser(int step1_blockSize, int step1_windowSize, int step1_maxMatch,
                             double step1_threDist, int step1_spdupFactor,
                             int step2_blockSize, int step2_windowSize, int step2_maxMatch,
                             double step2_threDist, int step2_spdupFactor,
                             double kaiserBeta, double lambda2d, double lambda3d)
    : m_step1BlockSize(step1_blockSize)
    , m_step1WindowSize(step1_windowSize)
    , m_step1MaxMatch(step1_maxMatch)
    , m_step1ThreDist(step1_threDist)
    , m_step1SpdupFactor(step1_spdupFactor)
    , m_step2BlockSize(step2_blockSize)
    , m_step2WindowSize(step2_windowSize)
    , m_step2MaxMatch(step2_maxMatch)
    , m_step2ThreDist(step2_threDist)
    , m_step2SpdupFactor(step2_spdupFactor)
    , m_kaiserBeta(kaiserBeta)
    , m_lambda2d(lambda2d)
    , m_lambda3d(lambda3d)
    , m_sigma(25.0)
{
    // Pre-warm cosine tables for expected group sizes
    DCTCosineTable::instance().getCosTable(step1_maxMatch);
    DCTCosineTable::instance().getCosTable(step2_maxMatch);
}

// ======================== Public Methods ========================

cv::Mat BM3D_Denoiser::denoise(const cv::Mat& input, float sigma) {
    m_sigma = static_cast<double>(sigma);

    // Convert input to CV_64F if necessary
    cv::Mat noisyImg;
    if (input.type() == CV_8U) {
        input.convertTo(noisyImg, CV_64F);
    } else if (input.type() == CV_64F) {
        noisyImg = input.clone();
    } else {
        input.convertTo(noisyImg, CV_64F);
    }

    // Step 1: Basic estimate
    cv::Mat basicImg = step1(noisyImg);

    // Step 2: Final estimate
    cv::Mat finalImg = step2(basicImg, noisyImg);

    // Normalize and convert to CV_8U
    cv::Mat output;
    cv::normalize(finalImg, output, 0, 255, cv::NORM_MINMAX);
    output.convertTo(output, CV_8U);

    return output;
}

cv::Mat BM3D_Denoiser::getBasicEstimate(const cv::Mat& input, float sigma) {
    m_sigma = static_cast<double>(sigma);

    cv::Mat noisyImg;
    if (input.type() == CV_8U) {
        input.convertTo(noisyImg, CV_64F);
    } else if (input.type() == CV_64F) {
        noisyImg = input.clone();
    } else {
        input.convertTo(noisyImg, CV_64F);
    }

    return step1(noisyImg);
}

// ======================== Helper Methods ========================

double BM3D_Denoiser::besselI0(double x) {
    // Compute zeroth-order modified Bessel function of the first kind
    // Using polynomial approximation for efficiency
    double ax = std::abs(x);
    double y;

    if (ax < 3.75) {
        y = x / 3.75;
        y = y * y;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
               + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    } else {
        y = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + y * (0.01328592
               + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
               + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633
               + y * 0.00392377))))))));
    }
}

cv::Mat BM3D_Denoiser::generateKaiserWindow(int size, double beta) {
    cv::Mat window(size, size, CV_64F);

    // Generate 1D Kaiser window
    std::vector<double> kaiser1D(size);
    double alpha = (size - 1) / 2.0;
    double denominator = besselI0(beta);

    for (int i = 0; i < size; ++i) {
        double ratio = (i - alpha) / alpha;
        double arg = beta * std::sqrt(1.0 - ratio * ratio);
        kaiser1D[i] = besselI0(arg) / denominator;
    }

    // Create 2D Kaiser window as outer product
    for (int i = 0; i < size; ++i) {
        double* rowPtr = window.ptr<double>(i);
        for (int j = 0; j < size; ++j) {
            rowPtr[j] = kaiser1D[i] * kaiser1D[j];
        }
    }

    return window;
}

cv::Mat BM3D_Denoiser::dct2D_ortho(const cv::Mat& input) {
    cv::Mat output;
    cv::dct(input, output);

    // Apply orthonormal scaling
    // OpenCV uses non-orthogonal DCT, we need to scale
    int rows = input.rows;
    int cols = input.cols;

    // Scale factors for orthonormal DCT
    double scale0_row = 1.0 / std::sqrt(static_cast<double>(rows));
    double scale1_row = std::sqrt(2.0 / static_cast<double>(rows));
    double scale0_col = 1.0 / std::sqrt(static_cast<double>(cols));
    double scale1_col = std::sqrt(2.0 / static_cast<double>(cols));

    for (int i = 0; i < rows; ++i) {
        double* rowPtr = output.ptr<double>(i);
        double rowScale = (i == 0) ? scale0_row : scale1_row;
        for (int j = 0; j < cols; ++j) {
            double colScale = (j == 0) ? scale0_col : scale1_col;
            rowPtr[j] *= rowScale * colScale;
        }
    }

    return output;
}

cv::Mat BM3D_Denoiser::idct2D_ortho(const cv::Mat& input) {
    cv::Mat scaled = input.clone();
    int rows = input.rows;
    int cols = input.cols;

    // Inverse scaling for orthonormal DCT
    double scale0_row = std::sqrt(static_cast<double>(rows));
    double scale1_row = std::sqrt(static_cast<double>(rows) / 2.0);
    double scale0_col = std::sqrt(static_cast<double>(cols));
    double scale1_col = std::sqrt(static_cast<double>(cols) / 2.0);

    for (int i = 0; i < rows; ++i) {
        double* rowPtr = scaled.ptr<double>(i);
        double rowScale = (i == 0) ? scale0_row : scale1_row;
        for (int j = 0; j < cols; ++j) {
            double colScale = (j == 0) ? scale0_col : scale1_col;
            rowPtr[j] *= rowScale * colScale;
        }
    }

    cv::Mat output;
    cv::idct(scaled, output);

    return output;
}

void BM3D_Denoiser::idct2D_ortho_buffer(const cv::Mat& input, cv::Mat& scaledBuffer, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;

    // Ensure buffers are properly sized
    if (scaledBuffer.rows != rows || scaledBuffer.cols != cols || scaledBuffer.type() != CV_64F) {
        scaledBuffer.create(rows, cols, CV_64F);
    }

    // Inverse scaling for orthonormal DCT
    double scale0_row = std::sqrt(static_cast<double>(rows));
    double scale1_row = std::sqrt(static_cast<double>(rows) / 2.0);
    double scale0_col = std::sqrt(static_cast<double>(cols));
    double scale1_col = std::sqrt(static_cast<double>(cols) / 2.0);

    for (int i = 0; i < rows; ++i) {
        const double* inPtr = input.ptr<double>(i);
        double* outPtr = scaledBuffer.ptr<double>(i);
        double rowScale = (i == 0) ? scale0_row : scale1_row;
        for (int j = 0; j < cols; ++j) {
            double colScale = (j == 0) ? scale0_col : scale1_col;
            outPtr[j] = inPtr[j] * rowScale * colScale;
        }
    }

    cv::idct(scaledBuffer, output);
}

// Fast 1D DCT using pre-computed cosine table (NO mutex locking - table passed as parameter)
void BM3D_Denoiser::fast_dct1d(const double* src, double* dst, int n, const std::vector<double>& cosTable) {
    double scale0 = 1.0 / std::sqrt(static_cast<double>(n));
    double scale1 = std::sqrt(2.0 / static_cast<double>(n));
    
    for (int k = 0; k < n; ++k) {
        double sum = 0.0;
        const double* cosRow = &cosTable[k * n];
        for (int i = 0; i < n; ++i) {
            sum += src[i] * cosRow[i];
        }
        dst[k] = sum * ((k == 0) ? scale0 : scale1);
    }
}

// Fast 1D IDCT using pre-computed cosine table (NO mutex locking - table passed as parameter)
void BM3D_Denoiser::fast_idct1d(const double* src, double* dst, int n, const std::vector<double>& cosTable) {
    double scale0 = 1.0 / std::sqrt(static_cast<double>(n));
    double scale1 = std::sqrt(2.0 / static_cast<double>(n));

    for (int i = 0; i < n; ++i) {
        double sum = src[0] * scale0;
        for (int k = 1; k < n; ++k) {
            sum += src[k] * scale1 * cosTable[k * n + i];
        }
        dst[i] = sum;
    }
}

std::vector<std::vector<cv::Mat>> BM3D_Denoiser::preDCT(const cv::Mat& img, int blockSize) {
    int numRows = img.rows - blockSize;
    int numCols = img.cols - blockSize;

    std::vector<std::vector<cv::Mat>> blockDCT_all(numRows, std::vector<cv::Mat>(numCols));

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            cv::Mat block = img(cv::Rect(j, i, blockSize, blockSize)).clone();
            blockDCT_all[i][j] = dct2D_ortho(block);
        }
    }

    return blockDCT_all;
}

void BM3D_Denoiser::searchWindow(int imgRows, int imgCols, const cv::Point& refPoint,
                                  int blockSize, int windowSize, cv::Vec4i& margin) {
    // Calculate search window boundaries
    int halfDiff = (blockSize - windowSize) / 2;

    margin[0] = std::max(0, refPoint.y + halfDiff); // top-left y (row)
    margin[1] = std::max(0, refPoint.x + halfDiff); // top-left x (col)
    margin[2] = margin[0] + windowSize;              // bottom-right y
    margin[3] = margin[1] + windowSize;              // bottom-right x

    // Adjust for image boundaries
    if (margin[2] >= imgRows) {
        margin[2] = imgRows - 1;
        margin[0] = margin[2] - windowSize;
    }
    if (margin[3] >= imgCols) {
        margin[3] = imgCols - 1;
        margin[1] = margin[3] - windowSize;
    }
}

// ======================== Step 1: Basic Estimate ========================

cv::Mat BM3D_Denoiser::step1(const cv::Mat& noisyImg) {
    int blockSize = m_step1BlockSize;
    double threDist = m_step1ThreDist;
    int maxMatch = m_step1MaxMatch;
    int windowSize = m_step1WindowSize;
    int spdupFactor = m_step1SpdupFactor;

    // Initialize output images and weights
    cv::Mat basicImg = cv::Mat::zeros(noisyImg.size(), CV_64F);
    cv::Mat basicWeight = cv::Mat::zeros(noisyImg.size(), CV_64F);
    cv::Mat basicKaiser = generateKaiserWindow(blockSize, m_kaiserBeta);

    // Pre-compute DCT for all blocks
    std::vector<std::vector<cv::Mat>> blockDCT_all = preDCT(noisyImg, blockSize);

    int numRefI = static_cast<int>((noisyImg.rows - blockSize) / spdupFactor) + 2;
    int numRefJ = static_cast<int>((noisyImg.cols - blockSize) / spdupFactor) + 2;

    // Process reference blocks
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ri = 0; ri < numRefI; ++ri) {
        for (int rj = 0; rj < numRefJ; ++rj) {
            int refY = std::min(spdupFactor * ri, noisyImg.rows - blockSize - 1);
            int refX = std::min(spdupFactor * rj, noisyImg.cols - blockSize - 1);
            cv::Point refPoint(refX, refY);

            std::vector<cv::Point> blockPos;
            std::vector<cv::Mat> blockGroup;

            step1_grouping(noisyImg, refPoint, blockDCT_all, blockSize,
                          threDist, maxMatch, windowSize, blockPos, blockGroup);

            int nonzeroCnt = step1_3DFiltering(blockGroup);

#pragma omp critical
            {
                step1_aggregation(blockGroup, blockPos, basicImg, basicWeight,
                                 basicKaiser, nonzeroCnt);
            }
        }
    }

    // Avoid division by zero
    for (int i = 0; i < basicWeight.rows; ++i) {
        double* weightPtr = basicWeight.ptr<double>(i);
        for (int j = 0; j < basicWeight.cols; ++j) {
            if (weightPtr[j] == 0.0) {
                weightPtr[j] = 1.0;
            }
        }
    }

    // Normalize
    cv::divide(basicImg, basicWeight, basicImg);

    return basicImg;
}

void BM3D_Denoiser::step1_grouping(const cv::Mat& noisyImg, const cv::Point& refPoint,
                                    const std::vector<std::vector<cv::Mat>>& blockDCT_all,
                                    int blockSize, double threDist, int maxMatch,
                                    int windowSize, std::vector<cv::Point>& blockPos,
                                    std::vector<cv::Mat>& blockGroup) {
    cv::Vec4i margin;
    searchWindow(noisyImg.rows, noisyImg.cols, refPoint, blockSize, windowSize, margin);

    int blockNumSearched = (windowSize - blockSize + 1) * (windowSize - blockSize + 1);

    std::vector<cv::Point> tempBlockPos(blockNumSearched);
    std::vector<cv::Mat> tempBlockGroup(blockNumSearched);
    std::vector<double> dist(blockNumSearched);

    const cv::Mat& refDCT = blockDCT_all[refPoint.y][refPoint.x];
    int matchCnt = 0;

    // Pre-compute threshold parameters for distance calculation
    bool useThreshold = (m_sigma > 40);
    double threValue = m_lambda2d * m_sigma;

    // Block searching
    for (int i = 0; i < windowSize - blockSize + 1; ++i) {
        for (int j = 0; j < windowSize - blockSize + 1; ++j) {
            int searchY = margin[0] + i;
            int searchX = margin[1] + j;

            if (searchY < 0 || searchY >= static_cast<int>(blockDCT_all.size()) ||
                searchX < 0 || searchX >= static_cast<int>(blockDCT_all[0].size())) {
                continue;
            }

            const cv::Mat& searchedDCT = blockDCT_all[searchY][searchX];
            double d = step1_computeDist_fast(refDCT, searchedDCT, blockSize, threValue, useThreshold);

            if (d < threDist) {
                tempBlockPos[matchCnt] = cv::Point(searchX, searchY);
                tempBlockGroup[matchCnt] = searchedDCT.clone();
                dist[matchCnt] = d;
                matchCnt++;
            }
        }
    }

    if (matchCnt <= maxMatch) {
        blockPos.assign(tempBlockPos.begin(), tempBlockPos.begin() + matchCnt);
        blockGroup.assign(tempBlockGroup.begin(), tempBlockGroup.begin() + matchCnt);
    } else {
        // Find indices of maxMatch smallest distances
        std::vector<int> indices(matchCnt);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + maxMatch, indices.end(),
                          [&dist](int a, int b) { return dist[a] < dist[b]; });

        blockPos.resize(maxMatch);
        blockGroup.resize(maxMatch);
        for (int k = 0; k < maxMatch; ++k) {
            blockPos[k] = tempBlockPos[indices[k]];
            blockGroup[k] = tempBlockGroup[indices[k]];
        }
    }
}

double BM3D_Denoiser::step1_computeDist_fast(const cv::Mat& blockDCT1, const cv::Mat& blockDCT2,
                                              int blockSize, double threValue, bool useThreshold) {
    double sumSq = 0.0;
    
    if (useThreshold) {
        // High noise case: apply soft thresholding before computing distance
        // Avoid allocation by computing inline
        for (int i = 0; i < blockSize; ++i) {
            const double* ptr1 = blockDCT1.ptr<double>(i);
            const double* ptr2 = blockDCT2.ptr<double>(i);
            for (int j = 0; j < blockSize; ++j) {
                double v1 = (std::abs(ptr1[j]) < threValue) ? 0.0 : ptr1[j];
                double v2 = (std::abs(ptr2[j]) < threValue) ? 0.0 : ptr2[j];
                double diff = v1 - v2;
                sumSq += diff * diff;
            }
        }
    } else {
        // Normal case: direct difference
        for (int i = 0; i < blockSize; ++i) {
            const double* ptr1 = blockDCT1.ptr<double>(i);
            const double* ptr2 = blockDCT2.ptr<double>(i);
            for (int j = 0; j < blockSize; ++j) {
                double diff = ptr1[j] - ptr2[j];
                sumSq += diff * diff;
            }
        }
    }

    return sumSq / (blockSize * blockSize);
}

int BM3D_Denoiser::step1_3DFiltering(std::vector<cv::Mat>& blockGroup) {
    if (blockGroup.empty()) return 0;

    double threValue = m_lambda3d * m_sigma;
    int nonzeroCnt = 0;
    int blockSize = blockGroup[0].rows;
    int numBlocks = static_cast<int>(blockGroup.size());

    // KEY OPTIMIZATION: Fetch cosine table ONCE before the hot loop (single mutex lock)
    const std::vector<double>& cosTable = DCTCosineTable::instance().getCosTable(numBlocks);

    // Pre-allocate buffers outside the loop to avoid heap allocations
    std::vector<double> buf_src(numBlocks);
    std::vector<double> buf_dct(numBlocks);

    // Apply 1D DCT along the third dimension, hard threshold, then inverse
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            // 1. Extract data using ptr for better performance
            for (int k = 0; k < numBlocks; ++k) {
                buf_src[k] = blockGroup[k].ptr<double>(i)[j];
            }

            // 2. Fast DCT (pass cosTable directly - NO mutex lock)
            fast_dct1d(buf_src.data(), buf_dct.data(), numBlocks, cosTable);

            // 3. Hard thresholding
            for (int k = 0; k < numBlocks; ++k) {
                if (std::abs(buf_dct[k]) < threValue) {
                    buf_dct[k] = 0.0;
                } else {
                    nonzeroCnt++;
                }
            }

            // 4. Fast IDCT (pass cosTable directly - NO mutex lock)
            fast_idct1d(buf_dct.data(), buf_src.data(), numBlocks, cosTable);

            // 5. Put back
            for (int k = 0; k < numBlocks; ++k) {
                blockGroup[k].ptr<double>(i)[j] = buf_src[k];
            }
        }
    }

    return nonzeroCnt;
}

void BM3D_Denoiser::step1_aggregation(const std::vector<cv::Mat>& blockGroup,
                                       const std::vector<cv::Point>& blockPos,
                                       cv::Mat& basicImg, cv::Mat& basicWeight,
                                       const cv::Mat& kaiser, int nonzeroCnt) {
    if (blockGroup.empty()) return;

    int blockSize = blockGroup[0].rows;
    double blockWeight;

    if (nonzeroCnt < 1) {
        blockWeight = 1.0;
    } else {
        blockWeight = 1.0 / (m_sigma * m_sigma * nonzeroCnt);
    }

    // Pre-compute weighted kaiser
    cv::Mat weightedKaiser = blockWeight * kaiser;

    // Pre-allocate buffers for IDCT
    cv::Mat scaledBuffer(blockSize, blockSize, CV_64F);
    cv::Mat reconstructed(blockSize, blockSize, CV_64F);

    for (size_t i = 0; i < blockPos.size(); ++i) {
        idct2D_ortho_buffer(blockGroup[i], scaledBuffer, reconstructed);

        int startY = blockPos[i].y;
        int startX = blockPos[i].x;

        for (int bi = 0; bi < blockSize; ++bi) {
            double* imgPtr = basicImg.ptr<double>(startY + bi) + startX;
            double* weightPtr = basicWeight.ptr<double>(startY + bi) + startX;
            const double* reconPtr = reconstructed.ptr<double>(bi);
            const double* kaiserPtr = weightedKaiser.ptr<double>(bi);

            for (int bj = 0; bj < blockSize; ++bj) {
                imgPtr[bj] += kaiserPtr[bj] * reconPtr[bj];
                weightPtr[bj] += kaiserPtr[bj];
            }
        }
    }
}

// ======================== Step 2: Final Estimate ========================

cv::Mat BM3D_Denoiser::step2(const cv::Mat& basicImg, const cv::Mat& noisyImg) {
    int blockSize = m_step2BlockSize;
    double threDist = m_step2ThreDist;
    int maxMatch = m_step2MaxMatch;
    int windowSize = m_step2WindowSize;
    int spdupFactor = m_step2SpdupFactor;

    // Initialize output images and weights
    cv::Mat finalImg = cv::Mat::zeros(basicImg.size(), CV_64F);
    cv::Mat finalWeight = cv::Mat::zeros(basicImg.size(), CV_64F);
    cv::Mat finalKaiser = generateKaiserWindow(blockSize, m_kaiserBeta);

    // Pre-compute DCT for all blocks
    std::vector<std::vector<cv::Mat>> blockDCT_basic = preDCT(basicImg, blockSize);
    std::vector<std::vector<cv::Mat>> blockDCT_noisy = preDCT(noisyImg, blockSize);

    int numRefI = static_cast<int>((basicImg.rows - blockSize) / spdupFactor) + 2;
    int numRefJ = static_cast<int>((basicImg.cols - blockSize) / spdupFactor) + 2;

    // Process reference blocks
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ri = 0; ri < numRefI; ++ri) {
        for (int rj = 0; rj < numRefJ; ++rj) {
            int refY = std::min(spdupFactor * ri, basicImg.rows - blockSize - 1);
            int refX = std::min(spdupFactor * rj, basicImg.cols - blockSize - 1);
            cv::Point refPoint(refX, refY);

            std::vector<cv::Point> blockPos;
            std::vector<cv::Mat> blockGroup_basic;
            std::vector<cv::Mat> blockGroup_noisy;

            step2_grouping(basicImg, noisyImg, refPoint, blockSize, threDist, maxMatch,
                          windowSize, blockDCT_basic, blockDCT_noisy,
                          blockPos, blockGroup_basic, blockGroup_noisy);

            double wienerWeight = step2_3DFiltering(blockGroup_basic, blockGroup_noisy);

#pragma omp critical
            {
                step2_aggregation(blockGroup_noisy, wienerWeight, blockPos,
                                 finalImg, finalWeight, finalKaiser);
            }
        }
    }

    // Avoid division by zero
    for (int i = 0; i < finalWeight.rows; ++i) {
        double* weightPtr = finalWeight.ptr<double>(i);
        for (int j = 0; j < finalWeight.cols; ++j) {
            if (weightPtr[j] == 0.0) {
                weightPtr[j] = 1.0;
            }
        }
    }

    // Normalize
    cv::divide(finalImg, finalWeight, finalImg);

    return finalImg;
}

void BM3D_Denoiser::step2_grouping(const cv::Mat& basicImg, const cv::Mat& noisyImg,
                                    const cv::Point& refPoint, int blockSize,
                                    double threDist, int maxMatch, int windowSize,
                                    const std::vector<std::vector<cv::Mat>>& blockDCT_basic,
                                    const std::vector<std::vector<cv::Mat>>& blockDCT_noisy,
                                    std::vector<cv::Point>& blockPos,
                                    std::vector<cv::Mat>& blockGroup_basic,
                                    std::vector<cv::Mat>& blockGroup_noisy) {
    (void)noisyImg; // Suppress unused parameter warning
    
    cv::Vec4i margin;
    searchWindow(basicImg.rows, basicImg.cols, refPoint, blockSize, windowSize, margin);

    int blockNumSearched = (windowSize - blockSize + 1) * (windowSize - blockSize + 1);

    std::vector<cv::Point> tempBlockPos(blockNumSearched);
    std::vector<double> dist(blockNumSearched);
    int matchCnt = 0;

    // Block searching using spatial domain distance
    for (int i = 0; i < windowSize - blockSize + 1; ++i) {
        for (int j = 0; j < windowSize - blockSize + 1; ++j) {
            int searchY = margin[0] + i;
            int searchX = margin[1] + j;
            cv::Point searchedPoint(searchX, searchY);

            if (searchY < 0 || searchY >= static_cast<int>(blockDCT_basic.size()) ||
                searchX < 0 || searchX >= static_cast<int>(blockDCT_basic[0].size())) {
                continue;
            }

            double d = step2_computeDist(basicImg, refPoint, searchedPoint, blockSize);

            if (d < threDist) {
                tempBlockPos[matchCnt] = searchedPoint;
                dist[matchCnt] = d;
                matchCnt++;
            }
        }
    }

    std::vector<cv::Point> selectedPos;
    if (matchCnt <= maxMatch) {
        selectedPos.assign(tempBlockPos.begin(), tempBlockPos.begin() + matchCnt);
    } else {
        // Find indices of maxMatch smallest distances
        std::vector<int> indices(matchCnt);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + maxMatch, indices.end(),
                          [&dist](int a, int b) { return dist[a] < dist[b]; });

        selectedPos.resize(maxMatch);
        for (int k = 0; k < maxMatch; ++k) {
            selectedPos[k] = tempBlockPos[indices[k]];
        }
    }

    // Populate block groups
    blockPos = selectedPos;
    blockGroup_basic.resize(selectedPos.size());
    blockGroup_noisy.resize(selectedPos.size());

    for (size_t i = 0; i < selectedPos.size(); ++i) {
        int y = selectedPos[i].y;
        int x = selectedPos[i].x;
        blockGroup_basic[i] = blockDCT_basic[y][x].clone();
        blockGroup_noisy[i] = blockDCT_noisy[y][x].clone();
    }
}

double BM3D_Denoiser::step2_computeDist(const cv::Mat& img, const cv::Point& point1,
                                         const cv::Point& point2, int blockSize) {
    // Optimized distance computation without creating temporary cv::Mat
    double sumSq = 0.0;
    
    for (int i = 0; i < blockSize; ++i) {
        const double* row1 = img.ptr<double>(point1.y + i) + point1.x;
        const double* row2 = img.ptr<double>(point2.y + i) + point2.x;
        for (int j = 0; j < blockSize; ++j) {
            double diff = row1[j] - row2[j];
            sumSq += diff * diff;
        }
    }
    
    return sumSq / (blockSize * blockSize);
}

double BM3D_Denoiser::step2_3DFiltering(const std::vector<cv::Mat>& blockGroup_basic,
                                         std::vector<cv::Mat>& blockGroup_noisy) {
    if (blockGroup_noisy.empty()) return 1.0;

    double weight = 0.0;
    int blockSize = blockGroup_noisy[0].rows;
    int numBlocks = static_cast<int>(blockGroup_noisy.size());
    double coef = 1.0 / numBlocks;
    double sigma2 = m_sigma * m_sigma;

    // KEY OPTIMIZATION: Fetch cosine table ONCE before the hot loop (single mutex lock)
    const std::vector<double>& cosTable = DCTCosineTable::instance().getCosTable(numBlocks);

    // Pre-allocate buffers outside the loop
    std::vector<double> vecBasic(numBlocks);
    std::vector<double> vecNoisy(numBlocks);
    std::vector<double> dctBasic(numBlocks);
    std::vector<double> dctNoisy(numBlocks);

    // Apply Wiener filtering
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            // Extract vectors along third dimension using ptr
            for (int k = 0; k < numBlocks; ++k) {
                vecBasic[k] = blockGroup_basic[k].ptr<double>(i)[j];
                vecNoisy[k] = blockGroup_noisy[k].ptr<double>(i)[j];
            }

            // Fast 1D DCT (pass cosTable directly - NO mutex lock)
            fast_dct1d(vecBasic.data(), dctBasic.data(), numBlocks, cosTable);
            fast_dct1d(vecNoisy.data(), dctNoisy.data(), numBlocks, cosTable);

            // Wiener filtering
            for (int k = 0; k < numBlocks; ++k) {
                double vecValue = dctBasic[k] * dctBasic[k] * coef;
                double wienerCoef = vecValue / (vecValue + sigma2);
                dctNoisy[k] *= wienerCoef;
                weight += wienerCoef;
            }

            // Fast Inverse 1D DCT (pass cosTable directly - NO mutex lock)
            fast_idct1d(dctNoisy.data(), vecNoisy.data(), numBlocks, cosTable);

            // Put back using ptr
            for (int k = 0; k < numBlocks; ++k) {
                blockGroup_noisy[k].ptr<double>(i)[j] = vecNoisy[k];
            }
        }
    }

    if (weight > 0) {
        return 1.0 / (sigma2 * weight);
    } else {
        return 1.0;
    }
}

void BM3D_Denoiser::step2_aggregation(const std::vector<cv::Mat>& blockGroup_noisy,
                                       double wienerWeight,
                                       const std::vector<cv::Point>& blockPos,
                                       cv::Mat& finalImg, cv::Mat& finalWeight,
                                       const cv::Mat& kaiser) {
    if (blockGroup_noisy.empty()) return;

    int blockSize = blockGroup_noisy[0].rows;
    cv::Mat weightedKaiser = wienerWeight * kaiser;

    // Pre-allocate buffers for IDCT
    cv::Mat scaledBuffer(blockSize, blockSize, CV_64F);
    cv::Mat reconstructed(blockSize, blockSize, CV_64F);

    for (size_t i = 0; i < blockPos.size(); ++i) {
        idct2D_ortho_buffer(blockGroup_noisy[i], scaledBuffer, reconstructed);

        int startY = blockPos[i].y;
        int startX = blockPos[i].x;

        for (int bi = 0; bi < blockSize; ++bi) {
            double* imgPtr = finalImg.ptr<double>(startY + bi) + startX;
            double* weightPtr = finalWeight.ptr<double>(startY + bi) + startX;
            const double* reconPtr = reconstructed.ptr<double>(bi);
            const double* kaiserPtr = weightedKaiser.ptr<double>(bi);

            for (int bj = 0; bj < blockSize; ++bj) {
                imgPtr[bj] += kaiserPtr[bj] * reconPtr[bj];
                weightPtr[bj] += kaiserPtr[bj];
            }
        }
    }
}
