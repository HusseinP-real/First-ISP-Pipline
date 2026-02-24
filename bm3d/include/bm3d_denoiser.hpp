/**
 * @file bm3d_denoiser.hpp
 * @brief BM3D (Block-Matching and 3D filtering) Image Denoising Algorithm
 * 
 * A C++ implementation of the BM3D denoising algorithm based on:
 * [1] Image denoising by sparse 3D transform-domain collaborative filtering
 * [2] An Analysis and Implementation of the BM3D Image Denoising Method
 */

#ifndef BM3D_DENOISER_HPP
#define BM3D_DENOISER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>
#include <mutex>

/**
 * @class DCTCosineTable
 * @brief Pre-computed cosine lookup table for DCT transforms (Singleton)
 * 
 * Caches cosine values to avoid repeated std::cos() calls in hot loops.
 * Thread-safe with mutex protection.
 */
class DCTCosineTable {
public:
    static DCTCosineTable& instance();
    
    /**
     * @brief Get or compute cosine table for given size
     * @param n Vector length
     * @return Reference to cached cosine table
     * @note Call this BEFORE entering hot loops to avoid mutex contention
     */
    const std::vector<double>& getCosTable(int n);

private:
    DCTCosineTable() = default;
    std::map<int, std::vector<double>> m_tables;
    std::mutex m_mutex;
};

/**
 * @class BM3D_Denoiser
 * @brief Block-Matching and 3D filtering denoiser
 * 
 * This class implements the BM3D algorithm for image denoising using
 * collaborative filtering in the transform domain.
 */
class BM3D_Denoiser {
public:
    /**
     * @brief Constructor with default parameters
     */
    BM3D_Denoiser();

    /**
     * @brief Constructor with custom parameters
     * @param step1_blockSize Block size for step 1 (default: 8)
     * @param step1_windowSize Search window size for step 1 (default: 39)
     * @param step1_maxMatch Maximum matched blocks for step 1 (default: 16)
     * @param step1_threDist Threshold distance for step 1 (default: 2500)
     * @param step1_spdupFactor Speed-up factor for step 1 (default: 3)
     * @param step2_blockSize Block size for step 2 (default: 8)
     * @param step2_windowSize Search window size for step 2 (default: 39)
     * @param step2_maxMatch Maximum matched blocks for step 2 (default: 32)
     * @param step2_threDist Threshold distance for step 2 (default: 400)
     * @param step2_spdupFactor Speed-up factor for step 2 (default: 3)
     * @param kaiserBeta Kaiser window beta parameter (default: 2.0)
     * @param lambda2d Lambda for 2D thresholding (default: 2.0)
     * @param lambda3d Lambda for 3D thresholding (default: 2.7)
     */
    BM3D_Denoiser(int step1_blockSize, int step1_windowSize, int step1_maxMatch,
                  double step1_threDist, int step1_spdupFactor,
                  int step2_blockSize, int step2_windowSize, int step2_maxMatch,
                  double step2_threDist, int step2_spdupFactor,
                  double kaiserBeta, double lambda2d, double lambda3d);

    /**
     * @brief Denoise an input image
     * @param input Input grayscale image (CV_8U or CV_64F)
     * @param sigma Noise standard deviation
     * @return Denoised image (CV_8U)
     */
    cv::Mat denoise(const cv::Mat& input, float sigma);

    /**
     * @brief Get the basic estimate (after Step 1 only)
     * @param input Input grayscale image
     * @param sigma Noise standard deviation
     * @return Basic estimate image (CV_64F)
     */
    cv::Mat getBasicEstimate(const cv::Mat& input, float sigma);

private:
    // Step 1 parameters
    int m_step1BlockSize;
    int m_step1WindowSize;
    int m_step1MaxMatch;
    double m_step1ThreDist;
    int m_step1SpdupFactor;

    // Step 2 parameters
    int m_step2BlockSize;
    int m_step2WindowSize;
    int m_step2MaxMatch;
    double m_step2ThreDist;
    int m_step2SpdupFactor;

    // Common parameters
    double m_kaiserBeta;
    double m_lambda2d;
    double m_lambda3d;
    double m_sigma;

    // ======================== Helper Methods ========================

    /**
     * @brief Generate a 2D Kaiser window
     * @param size Window size
     * @param beta Kaiser beta parameter
     * @return 2D Kaiser window matrix (CV_64F)
     */
    cv::Mat generateKaiserWindow(int size, double beta);

    /**
     * @brief Compute the zeroth-order modified Bessel function of the first kind
     * @param x Input value
     * @return I0(x)
     */
    double besselI0(double x);

    /**
     * @brief Perform 2D orthonormal DCT
     * @param input Input block
     * @return DCT transformed block
     */
    cv::Mat dct2D_ortho(const cv::Mat& input);

    /**
     * @brief Perform inverse 2D orthonormal DCT
     * @param input DCT block
     * @return Inverse DCT transformed block
     */
    cv::Mat idct2D_ortho(const cv::Mat& input);

    /**
     * @brief Perform inverse 2D orthonormal DCT (no allocation, reuses buffer)
     * @param input DCT block
     * @param scaledBuffer Pre-allocated buffer for scaling
     * @param output Pre-allocated output buffer
     */
    void idct2D_ortho_buffer(const cv::Mat& input, cv::Mat& scaledBuffer, cv::Mat& output);

    /**
     * @brief Fast 1D DCT with pre-computed cosine table (no allocation, no mutex)
     * @param src Source data pointer
     * @param dst Destination data pointer
     * @param n Vector length
     * @param cosTable Pre-fetched cosine table reference
     */
    void fast_dct1d(const double* src, double* dst, int n, const std::vector<double>& cosTable);

    /**
     * @brief Fast 1D IDCT with pre-computed cosine table (no allocation, no mutex)
     * @param src Source data pointer
     * @param dst Destination data pointer
     * @param n Vector length
     * @param cosTable Pre-fetched cosine table reference
     */
    void fast_idct1d(const double* src, double* dst, int n, const std::vector<double>& cosTable);

    /**
     * @brief Pre-compute DCT for all blocks in the image
     * @param img Input image
     * @param blockSize Block size
     * @return 4D array of DCT blocks (stored as vector of vector of cv::Mat)
     */
    std::vector<std::vector<cv::Mat>> preDCT(const cv::Mat& img, int blockSize);

    /**
     * @brief Find search window boundaries
     * @param imgRows Image height
     * @param imgCols Image width
     * @param refPoint Reference point coordinates
     * @param blockSize Block size
     * @param windowSize Search window size
     * @param margin Output: [top-left y, top-left x, bottom-right y, bottom-right x]
     */
    void searchWindow(int imgRows, int imgCols, const cv::Point& refPoint,
                      int blockSize, int windowSize, cv::Vec4i& margin);

    // ======================== Step 1: Basic Estimate ========================

    /**
     * @brief Execute BM3D Step 1 (basic estimate)
     * @param noisyImg Noisy input image
     * @return Basic estimate image
     */
    cv::Mat step1(const cv::Mat& noisyImg);

    /**
     * @brief Step 1 grouping - find similar blocks
     * @param noisyImg Noisy image
     * @param refPoint Reference block position
     * @param blockDCT_all Pre-computed DCT blocks
     * @param blockSize Block size
     * @param threDist Threshold distance
     * @param maxMatch Maximum matches
     * @param windowSize Search window size
     * @param blockPos Output: positions of matched blocks
     * @param blockGroup Output: DCT blocks of matched blocks
     */
    void step1_grouping(const cv::Mat& noisyImg, const cv::Point& refPoint,
                        const std::vector<std::vector<cv::Mat>>& blockDCT_all,
                        int blockSize, double threDist, int maxMatch, int windowSize,
                        std::vector<cv::Point>& blockPos,
                        std::vector<cv::Mat>& blockGroup);

    /**
     * @brief Step 1 compute distance between two DCT blocks (optimized, no allocation)
     * @param blockDCT1 First DCT block
     * @param blockDCT2 Second DCT block
     * @param blockSize Block size
     * @param threValue Threshold value (for high noise)
     * @param useThreshold Whether to apply thresholding
     * @return Normalized squared distance
     */
    double step1_computeDist_fast(const cv::Mat& blockDCT1, const cv::Mat& blockDCT2,
                                   int blockSize, double threValue, bool useThreshold);

    /**
     * @brief Step 1 3D filtering (hard thresholding) - optimized version
     * @param blockGroup Group of DCT blocks (modified in place)
     * @return Number of non-zero coefficients
     */
    int step1_3DFiltering(std::vector<cv::Mat>& blockGroup);

    /**
     * @brief Step 1 aggregation
     * @param blockGroup Filtered block group
     * @param blockPos Block positions
     * @param basicImg Output image (accumulated)
     * @param basicWeight Output weights (accumulated)
     * @param kaiser Kaiser window
     * @param nonzeroCnt Number of non-zero coefficients
     */
    void step1_aggregation(const std::vector<cv::Mat>& blockGroup,
                           const std::vector<cv::Point>& blockPos,
                           cv::Mat& basicImg, cv::Mat& basicWeight,
                           const cv::Mat& kaiser, int nonzeroCnt);

    // ======================== Step 2: Final Estimate ========================

    /**
     * @brief Execute BM3D Step 2 (final estimate)
     * @param basicImg Basic estimate from Step 1
     * @param noisyImg Original noisy image
     * @return Final denoised image
     */
    cv::Mat step2(const cv::Mat& basicImg, const cv::Mat& noisyImg);

    /**
     * @brief Step 2 grouping - find similar blocks using basic image
     * @param basicImg Basic estimate image
     * @param noisyImg Noisy image
     * @param refPoint Reference block position
     * @param blockSize Block size
     * @param threDist Threshold distance
     * @param maxMatch Maximum matches
     * @param windowSize Search window size
     * @param blockDCT_basic Pre-computed DCT of basic image
     * @param blockDCT_noisy Pre-computed DCT of noisy image
     * @param blockPos Output: positions of matched blocks
     * @param blockGroup_basic Output: DCT blocks from basic image
     * @param blockGroup_noisy Output: DCT blocks from noisy image
     */
    void step2_grouping(const cv::Mat& basicImg, const cv::Mat& noisyImg,
                        const cv::Point& refPoint, int blockSize,
                        double threDist, int maxMatch, int windowSize,
                        const std::vector<std::vector<cv::Mat>>& blockDCT_basic,
                        const std::vector<std::vector<cv::Mat>>& blockDCT_noisy,
                        std::vector<cv::Point>& blockPos,
                        std::vector<cv::Mat>& blockGroup_basic,
                        std::vector<cv::Mat>& blockGroup_noisy);

    /**
     * @brief Step 2 compute distance between two blocks in spatial domain (optimized)
     * @param img Input image
     * @param point1 First block position
     * @param point2 Second block position
     * @param blockSize Block size
     * @return Normalized squared distance
     */
    double step2_computeDist(const cv::Mat& img, const cv::Point& point1,
                             const cv::Point& point2, int blockSize);

    /**
     * @brief Step 2 3D Wiener filtering - optimized version
     * @param blockGroup_basic Basic image block group
     * @param blockGroup_noisy Noisy image block group (modified in place)
     * @return Wiener weight
     */
    double step2_3DFiltering(const std::vector<cv::Mat>& blockGroup_basic,
                             std::vector<cv::Mat>& blockGroup_noisy);

    /**
     * @brief Step 2 aggregation
     * @param blockGroup_noisy Filtered noisy block group
     * @param wienerWeight Wiener weight
     * @param blockPos Block positions
     * @param finalImg Output image (accumulated)
     * @param finalWeight Output weights (accumulated)
     * @param kaiser Kaiser window
     */
    void step2_aggregation(const std::vector<cv::Mat>& blockGroup_noisy,
                           double wienerWeight,
                           const std::vector<cv::Point>& blockPos,
                           cv::Mat& finalImg, cv::Mat& finalWeight,
                           const cv::Mat& kaiser);
};

#endif // BM3D_DENOISER_HPP
