#include "dct.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

static const int N = 8;
static const float PI = 3.14159265358979323846f;

// Precompute 8x8 DCT-II Matrix
// C[i][j] = alpha(i) * cos((2j+1)*i*PI / 2N)
static void computeDCTMatrix(float C[N][N], float Ct[N][N]) {
    float alpha0 = std::sqrt(1.0f / N);
    float alphaK = std::sqrt(2.0f / N);

    for (int i = 0; i < N; ++i) {
        float alpha = (i == 0) ? alpha0 : alphaK;
        for (int j = 0; j < N; ++j) {
            C[i][j] = alpha * std::cos((2 * j + 1) * i * PI / (2 * N));
            Ct[j][i] = C[i][j]; // Transpose
        }
    }
}

// 2D DCT: D = C * P * C^T
// P is 8x8 input patch, D is 8x8 output coefficients
static void forwardDCT(const float P[N][N], float D[N][N], const float C[N][N], const float Ct[N][N]) {
    float temp[N][N];
    
    // temp = C * P
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += C[i][k] * P[k][j];
            }
            temp[i][j] = sum;
        }
    }

    // D = temp * C^T
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += temp[i][k] * Ct[k][j];
            }
            D[i][j] = sum;
        }
    }
}

// 2D IDCT: P = C^T * D * C
static void inverseDCT(const float D[N][N], float P[N][N], const float C[N][N], const float Ct[N][N]) {
    float temp[N][N];

    // temp = C^T * D
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += Ct[i][k] * D[k][j];
            }
            temp[i][j] = sum;
        }
    }

    // P = temp * C
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += temp[i][k] * C[k][j];
            }
            P[i][j] = sum;
        }
    }
}

void denoiseDCT(const std::vector<float>& src, 
                std::vector<float>& dst, 
                int width, 
                int height, 
                float sigma,
                int step) {
    if (src.empty() || width <= 0 || height <= 0) return;

    dst.assign(src.size(), 0.0f);
    std::vector<float> countBuffer(src.size(), 0.0f);

    float C[N][N];
    float Ct[N][N];
    computeDCTMatrix(C, Ct);

    float threshold = 3.0f * sigma;

    // Sliding window
    for (int y = 0; y <= height - N; y += step) {
        for (int x = 0; x <= width - N; x += step) {
            
            // Extract patch P
            float P[N][N];
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    P[i][j] = src[(y + i) * width + (x + j)];
                }
            }

            // Forward DCT
            float D[N][N];
            forwardDCT(P, D, C, Ct);

            // Hard Thresholding
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (std::abs(D[i][j]) < threshold) {
                        D[i][j] = 0.0f;
                    }
                }
            }

            // Inverse DCT
            float P_rec[N][N];
            inverseDCT(D, P_rec, C, Ct);

            // Accumulate
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int idx = (y + i) * width + (x + j);
                    dst[idx] += P_rec[i][j];
                    countBuffer[idx] += 1.0f;
                }
            }
        }
    }

    // Average and handle boundaries (simply copy original for unprocessed edges)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (countBuffer[idx] > 0.0f) {
                dst[idx] /= countBuffer[idx];
            } else {
                dst[idx] = src[idx];
            }
             // Clip to valid range [0, 1] assuming float input is 0-1
             if (dst[idx] < 0.0f) dst[idx] = 0.0f;
             if (dst[idx] > 1.0f) dst[idx] = 1.0f;
        }
    }
}
