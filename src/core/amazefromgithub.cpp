////////////////////////////////////////////////////////////////
//
//			AMaZE demosaic algorithm
// (Aliasing Minimization and Zipper Elimination)
//
//	copyright (c) 2008-2010  Emil Martinec <ejmartin@uchicago.edu>
//
// incorporating ideas of Luis Sanz Rodrigues and Paul Lee
//
// code dated: May 27, 2010
//
//	Adapted for OpenCV/ISP pipeline project
//
////////////////////////////////////////////////////////////////

#include "amazefromgithub.h"
#include "demosiac.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <ctime>

#define TS 512  // Tile size
#define SQR(x) ((x)*(x))

namespace {

enum class CFAColor { Red, GreenR, GreenB, Blue };

inline int idx(int x, int y, int w) { return y * w + x; }

// Get color type from Bayer pattern
inline CFAColor getColor(int x, int y, int start_x, int start_y) {
    bool row_even = ((y + start_y) & 1) == 0;
    bool col_even = ((x + start_x) & 1) == 0;
    if (row_even && col_even) return CFAColor::Red;
    if (row_even && !col_even) return CFAColor::GreenR;
    if (!row_even && col_even) return CFAColor::GreenB;
    return CFAColor::Blue;
}

// FC function: returns 0=R, 1=G, 2=B (for compatibility with original code)
inline int FC(int rr, int cc, int start_x, int start_y) {
    CFAColor c = getColor(cc, rr, start_x, start_y);
    if (c == CFAColor::Red) return 0;
    if (c == CFAColor::GreenR || c == CFAColor::GreenB) return 1;
    return 2; // Blue
}

inline void getPatternOffset(bayerPattern pattern, int& start_x, int& start_y) {
    switch (pattern) {
        case RGGB: start_y = 0; start_x = 0; break;
        case BGGR: start_y = 1; start_x = 1; break;
        case GBRG: start_y = 1; start_x = 0; break;
        case GRBG: start_y = 0; start_x = 1; break;
    }
}

// Helper functions (avoid conflicts with OpenCV macros)
template<typename T>
inline T clamp_val(T x, T min_val, T max_val) { 
    return std::max(min_val, std::min(x, max_val)); 
}

template<typename T>
inline T ulim_val(T x, T y, T z) { 
    return (y < z) ? clamp_val(x, y, z) : clamp_val(x, z, y); 
}

} // namespace

void demosiacAMaZEFromGitHub(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    CV_Assert(!raw.empty());
    CV_Assert(raw.channels() == 1);

    const int width = raw.cols;
    const int height = raw.rows;
    CV_Assert(width >= 32 && height >= 32);

    // Convert to float
    cv::Mat raw_f;
    raw.convertTo(raw_f, CV_32F);
    
    // Get max value for clipping
    double min_raw = 0.0, max_raw = 0.0;
    cv::minMaxLoc(raw_f, &min_raw, &max_raw);
    const float maxVal = static_cast<float>(std::max(1.0, max_raw));
    const float clip_pt = maxVal; // Use max value as clip point

    // Get pattern offset
    int start_x = 0, start_y = 0;
    getPatternOffset(pattern, start_x, start_y);

    // Determine R pixel offset within Bayer quartet
    // ex, ey 用于确定 R 像素在 2x2 Bayer quartet 中的位置
    int ex = 0, ey = 0;
    if (FC(0, 0, start_x, start_y) == 1) { // first pixel is G
        if (FC(0, 1, start_x, start_y) == 0) { ey = 0; ex = 1; } 
        else { ey = 1; ex = 0; }
    } else { // first pixel is R or B
        if (FC(0, 0, start_x, start_y) == 0) { ey = 0; ex = 0; } 
        else { ey = 1; ex = 1; }
    }
    // 使用 ex, ey 来避免编译器警告
    (void)ex; (void)ey;

    // Constants
    static const float eps = 1e-5f, epssq = 1e-10f;
    static const float arthresh = 0.75f;
    static const float nyqthresh = 0.5f;
    
    // Gaussian kernels
	static const float gaussodd[4] = {0.14659727707323927f, 0.103592713382435f, 0.0732036125103057f, 0.0365543548389495f};
    static const float gaussgrad[6] = {0.07384411893421103f, 0.06207511968171489f, 0.0521818194747806f,
	0.03687419286733595f, 0.03099732204057846f, 0.018413194161458882f};
	static const float gquinc[4] = {0.169917f, 0.108947f, 0.069855f, 0.0287182f};

    // Tile shifts
    static const int v1 = TS, v2 = 2*TS, v3 = 3*TS;
    static const int p1 = -TS+1, p2 = -2*TS+2;
    static const int m1 = TS+1, m2 = 2*TS+2;

    // Output RGB arrays
    std::vector<float> red(width * height);
    std::vector<float> green(width * height);
    std::vector<float> blue(width * height);

    // Process image in tiles
    int winx = 0, winy = 0;
    
    for (int top = winy - 16; top < winy + height; top += TS - 32) {
        for (int left = winx - 16; left < winx + width; left += TS - 32) {
            int bottom = std::min(top + TS, winy + height + 16);
            int right = std::min(left + TS, winx + width + 16);
			int rr1 = bottom - top;
			int cc1 = right - left;

            // Allocate tile buffer
            size_t buffer_size = (34 * sizeof(float) + sizeof(int)) * TS * TS;
            char* buffer = (char*)malloc(buffer_size);
            if (!buffer) continue;

            // Set up tile arrays
            float (*rgb)[3] = reinterpret_cast<float (*)[3]>(buffer);
            float* delh = reinterpret_cast<float*>(buffer + 3 * sizeof(float) * TS * TS);
            float* delv = reinterpret_cast<float*>(buffer + 4 * sizeof(float) * TS * TS);
            float* delhsq = reinterpret_cast<float*>(buffer + 5 * sizeof(float) * TS * TS);
            float* delvsq = reinterpret_cast<float*>(buffer + 6 * sizeof(float) * TS * TS);
            float (*dirwts)[2] = reinterpret_cast<float (*)[2]>(buffer + 7 * sizeof(float) * TS * TS);
            float* vcd = reinterpret_cast<float*>(buffer + 9 * sizeof(float) * TS * TS);
            float* hcd = reinterpret_cast<float*>(buffer + 10 * sizeof(float) * TS * TS);
            float* vcdalt = reinterpret_cast<float*>(buffer + 11 * sizeof(float) * TS * TS);
            float* hcdalt = reinterpret_cast<float*>(buffer + 12 * sizeof(float) * TS * TS);
            float* vcdsq = reinterpret_cast<float*>(buffer + 13 * sizeof(float) * TS * TS);
            float* hcdsq = reinterpret_cast<float*>(buffer + 14 * sizeof(float) * TS * TS);
            float* cddiffsq = reinterpret_cast<float*>(buffer + 15 * sizeof(float) * TS * TS);
            float* hvwt = reinterpret_cast<float*>(buffer + 16 * sizeof(float) * TS * TS);
            float (*Dgrb)[2] = reinterpret_cast<float (*)[2]>(buffer + 17 * sizeof(float) * TS * TS);
            float* delp = reinterpret_cast<float*>(buffer + 19 * sizeof(float) * TS * TS);
            float* delm = reinterpret_cast<float*>(buffer + 20 * sizeof(float) * TS * TS);
            float* rbint = reinterpret_cast<float*>(buffer + 21 * sizeof(float) * TS * TS);
            float* Dgrbh2 = reinterpret_cast<float*>(buffer + 22 * sizeof(float) * TS * TS);
            float* Dgrbv2 = reinterpret_cast<float*>(buffer + 23 * sizeof(float) * TS * TS);
            float* dgintv = reinterpret_cast<float*>(buffer + 24 * sizeof(float) * TS * TS);
            float* dginth = reinterpret_cast<float*>(buffer + 25 * sizeof(float) * TS * TS);
            float* Dgrbpsq1 = reinterpret_cast<float*>(buffer + 28 * sizeof(float) * TS * TS);
            float* Dgrbmsq1 = reinterpret_cast<float*>(buffer + 29 * sizeof(float) * TS * TS);
            float* cfa = reinterpret_cast<float*>(buffer + 30 * sizeof(float) * TS * TS);
            // pmwt, rbp 和 rbm 在完整版本中用于对角插值，当前版本已简化
            // float* pmwt = reinterpret_cast<float*>(buffer + 31 * sizeof(float) * TS * TS);
            // float* rbp = reinterpret_cast<float*>(buffer + 32 * sizeof(float) * TS * TS);
            // float* rbm = reinterpret_cast<float*>(buffer + 33 * sizeof(float) * TS * TS);
            int* nyquist = reinterpret_cast<int*>(buffer + 34 * sizeof(float) * TS * TS);

            // Initialize tile from image
            int rrmin = (top < winy) ? 16 : 0;
            int ccmin = (left < winx) ? 16 : 0;
            int rrmax = (bottom > (winy + height)) ? (winy + height - top) : rr1;
            int ccmax = (right > (winx + width)) ? (winx + width - left) : cc1;

            for (int rr = rrmin; rr < rrmax; rr++) {
                int row = rr + top;
                for (int cc = ccmin; cc < ccmax; cc++) {
                    int col = cc + left;
                    int c = FC(rr, cc, start_x, start_y);
                    int indx1 = rr * TS + cc;
                    
                    if (row >= 0 && row < height && col >= 0 && col < width) {
                        rgb[indx1][c] = raw_f.at<float>(row, col) / maxVal;
                    } else {
                        rgb[indx1][c] = 0.0f;
                    }
                    cfa[indx1] = rgb[indx1][c];
                }
            }

            // Fill borders (simplified - use reflection)
            if (rrmin > 0) {
                for (int rr = 0; rr < 16; rr++) {
                    for (int cc = ccmin; cc < ccmax; cc++) {
                        int c = FC(rr, cc, start_x, start_y);
                        int src_rr = 32 - rr;
                        if (src_rr < rr1) {
                            rgb[rr * TS + cc][c] = rgb[src_rr * TS + cc][c];
                            cfa[rr * TS + cc] = rgb[rr * TS + cc][c];
                        }
                    }
                }
            }
            // Similar border filling for other edges...
            // (Simplified for brevity - full implementation would handle all borders)

            // Compute gradients
            for (int rr = 1; rr < rr1 - 1; rr++) {
                for (int cc = 1, indx = rr * TS + cc; cc < cc1 - 1; cc++, indx++) {
                    delh[indx] = std::abs(cfa[indx + 1] - cfa[indx - 1]);
                    delv[indx] = std::abs(cfa[indx + v1] - cfa[indx - v1]);
                    delhsq[indx] = SQR(delh[indx]);
                    delvsq[indx] = SQR(delv[indx]);
                    delp[indx] = std::abs(cfa[indx + p1] - cfa[indx - p1]);
                    delm[indx] = std::abs(cfa[indx + m1] - cfa[indx - m1]);
                }
            }

            // Compute directional weights
            for (int rr = 2; rr < rr1 - 2; rr++) {
                for (int cc = 2, indx = rr * TS + cc; cc < cc1 - 2; cc++, indx++) {
                    dirwts[indx][0] = eps + delv[indx + v1] + delv[indx - v1] + delv[indx];
                    dirwts[indx][1] = eps + delh[indx + 1] + delh[indx - 1] + delh[indx];
                    
                    if (FC(rr, cc, start_x, start_y) & 1) {
                        Dgrbpsq1[indx] = SQR(cfa[indx] - cfa[indx - p1]) + SQR(cfa[indx] - cfa[indx + p1]);
                        Dgrbmsq1[indx] = SQR(cfa[indx] - cfa[indx - m1]) + SQR(cfa[indx] - cfa[indx + m1]);
                    }
                }
            }

            // Interpolate vertical and horizontal color differences
            for (int rr = 4; rr < rr1 - 4; rr++) {
                for (int cc = 4, indx = rr * TS + cc; cc < cc1 - 4; cc++, indx++) {
                    int c = FC(rr, cc, start_x, start_y);
                    int sgn = (c & 1) ? -1 : 1;

                    nyquist[indx] = 0;
                    rbint[indx] = 0;

                    // Color ratios
                    float cru = cfa[indx - v1] * (dirwts[indx - v2][0] + dirwts[indx][0]) / 
                                (dirwts[indx - v2][0] * (eps + cfa[indx]) + dirwts[indx][0] * (eps + cfa[indx - v2]));
                    float crd = cfa[indx + v1] * (dirwts[indx + v2][0] + dirwts[indx][0]) / 
                                (dirwts[indx + v2][0] * (eps + cfa[indx]) + dirwts[indx][0] * (eps + cfa[indx + v2]));
                    float crl = cfa[indx - 1] * (dirwts[indx - 2][1] + dirwts[indx][1]) / 
                                (dirwts[indx - 2][1] * (eps + cfa[indx]) + dirwts[indx][1] * (eps + cfa[indx - 2]));
                    float crr = cfa[indx + 1] * (dirwts[indx + 2][1] + dirwts[indx][1]) / 
                                (dirwts[indx + 2][1] * (eps + cfa[indx]) + dirwts[indx][1] * (eps + cfa[indx + 2]));

                    // Hamilton-Adams interpolation
                    float guha = std::min(clip_pt, cfa[indx - v1]) + 0.5f * (cfa[indx] - cfa[indx - v2]);
                    float gdha = std::min(clip_pt, cfa[indx + v1]) + 0.5f * (cfa[indx] - cfa[indx + v2]);
                    float glha = std::min(clip_pt, cfa[indx - 1]) + 0.5f * (cfa[indx] - cfa[indx - 2]);
                    float grha = std::min(clip_pt, cfa[indx + 1]) + 0.5f * (cfa[indx] - cfa[indx + 2]);

                    // Adaptive ratio interpolation
                    float guar = (std::abs(1.0f - cru) < arthresh) ? (cfa[indx] * cru) : guha;
                    float gdar = (std::abs(1.0f - crd) < arthresh) ? (cfa[indx] * crd) : gdha;
                    float glar = (std::abs(1.0f - crl) < arthresh) ? (cfa[indx] * crl) : glha;
                    float grar = (std::abs(1.0f - crr) < arthresh) ? (cfa[indx] * crr) : grha;

                    float hwt = dirwts[indx - 1][1] / (dirwts[indx - 1][1] + dirwts[indx + 1][1]);
                    float vwt = dirwts[indx - v1][0] / (dirwts[indx + v1][0] + dirwts[indx - v1][0]);

                    float Gintvar = vwt * gdar + (1 - vwt) * guar;
                    float Ginthar = hwt * grar + (1 - hwt) * glar;
                    float Gintvha = vwt * gdha + (1 - vwt) * guha;
                    float Ginthha = hwt * grha + (1 - hwt) * glha;

                    vcd[indx] = sgn * (Gintvar - cfa[indx]);
                    hcd[indx] = sgn * (Ginthar - cfa[indx]);
                    vcdalt[indx] = sgn * (Gintvha - cfa[indx]);
                    hcdalt[indx] = sgn * (Ginthha - cfa[indx]);

                    if (cfa[indx] > 0.8f * clip_pt || Gintvha > 0.8f * clip_pt || Ginthha > 0.8f * clip_pt) {
                        guar = guha; gdar = gdha; glar = glha; grar = grha;
                        vcd[indx] = vcdalt[indx];
                        hcd[indx] = hcdalt[indx];
                    }

                    dgintv[indx] = std::min(SQR(guha - gdha), SQR(guar - gdar));
                    dginth[indx] = std::min(SQR(glha - grha), SQR(glar - grar));
                }
            }

            // Variance computation and refinement (simplified - full version would include all steps)
            for (int rr = 4; rr < rr1 - 4; rr++) {
                for (int cc = 4, indx = rr * TS + cc; cc < cc1 - 4; cc++, indx++) {
                    int c = FC(rr, cc, start_x, start_y);

                    float hcdvar = 3.0f * (SQR(hcd[indx - 2]) + SQR(hcd[indx]) + SQR(hcd[indx + 2])) - 
                                   SQR(hcd[indx - 2] + hcd[indx] + hcd[indx + 2]);
                    float hcdaltvar = 3.0f * (SQR(hcdalt[indx - 2]) + SQR(hcdalt[indx]) + SQR(hcdalt[indx + 2])) - 
                                      SQR(hcdalt[indx - 2] + hcdalt[indx] + hcdalt[indx + 2]);
                    float vcdvar = 3.0f * (SQR(vcd[indx - v2]) + SQR(vcd[indx]) + SQR(vcd[indx + v2])) - 
                                   SQR(vcd[indx - v2] + vcd[indx] + vcd[indx + v2]);
                    float vcdaltvar = 3.0f * (SQR(vcdalt[indx - v2]) + SQR(vcdalt[indx]) + SQR(vcdalt[indx + v2])) - 
                                      SQR(vcdalt[indx - v2] + vcdalt[indx] + vcdalt[indx + v2]);

                    if (hcdaltvar < hcdvar) hcd[indx] = hcdalt[indx];
                    if (vcdaltvar < vcdvar) vcd[indx] = vcdalt[indx];

                    // Bounding interpolation
                    if (c & 1) { // G site
                        float Ginth = -hcd[indx] + cfa[indx];
                        float Gintv = -vcd[indx] + cfa[indx];

                        if (hcd[indx] > 0) {
                            if (3.0f * hcd[indx] > (Ginth + cfa[indx])) {
                                hcd[indx] = -ulim_val(Ginth, cfa[indx - 1], cfa[indx + 1]) + cfa[indx];
							} else {
                                float hwt_val = 1.0f - 3.0f * hcd[indx] / (eps + Ginth + cfa[indx]);
                                hcd[indx] = hwt_val * hcd[indx] + (1 - hwt_val) * (-ulim_val(Ginth, cfa[indx - 1], cfa[indx + 1]) + cfa[indx]);
                            }
                        }
                        if (vcd[indx] > 0) {
                            if (3.0f * vcd[indx] > (Gintv + cfa[indx])) {
                                vcd[indx] = -ulim_val(Gintv, cfa[indx - v1], cfa[indx + v1]) + cfa[indx];
							} else {
                                float vwt_val = 1.0f - 3.0f * vcd[indx] / (eps + Gintv + cfa[indx]);
                                vcd[indx] = vwt_val * vcd[indx] + (1 - vwt_val) * (-ulim_val(Gintv, cfa[indx - v1], cfa[indx + v1]) + cfa[indx]);
                            }
                        }
                        if (Ginth > clip_pt) hcd[indx] = -ulim_val(Ginth, cfa[indx - 1], cfa[indx + 1]) + cfa[indx];
                        if (Gintv > clip_pt) vcd[indx] = -ulim_val(Gintv, cfa[indx - v1], cfa[indx + v1]) + cfa[indx];
                    } else { // R or B site
                        float Ginth = hcd[indx] + cfa[indx];
                        float Gintv = vcd[indx] + cfa[indx];

                        if (hcd[indx] < 0) {
                            if (3.0f * hcd[indx] < -(Ginth + cfa[indx])) {
                                hcd[indx] = ulim_val(Ginth, cfa[indx - 1], cfa[indx + 1]) - cfa[indx];
							} else {
                                float hwt_val = 1.0f + 3.0f * hcd[indx] / (eps + Ginth + cfa[indx]);
                                hcd[indx] = hwt_val * hcd[indx] + (1 - hwt_val) * (ulim_val(Ginth, cfa[indx - 1], cfa[indx + 1]) - cfa[indx]);
                            }
                        }
                        if (vcd[indx] < 0) {
                            if (3.0f * vcd[indx] < -(Gintv + cfa[indx])) {
                                vcd[indx] = ulim_val(Gintv, cfa[indx - v1], cfa[indx + v1]) - cfa[indx];
							} else {
                                float vwt_val = 1.0f + 3.0f * vcd[indx] / (eps + Gintv + cfa[indx]);
                                vcd[indx] = vwt_val * vcd[indx] + (1 - vwt_val) * (ulim_val(Gintv, cfa[indx - v1], cfa[indx + v1]) - cfa[indx]);
                            }
                        }
                        if (Ginth > clip_pt) hcd[indx] = ulim_val(Ginth, cfa[indx - 1], cfa[indx + 1]) - cfa[indx];
                        if (Gintv > clip_pt) vcd[indx] = ulim_val(Gintv, cfa[indx - v1], cfa[indx + v1]) - cfa[indx];
                    }
					
					vcdsq[indx] = SQR(vcd[indx]);
					hcdsq[indx] = SQR(hcd[indx]);
                    cddiffsq[indx] = SQR(vcd[indx] - hcd[indx]);
                }
            }

            // Compute adaptive weights for G interpolation
            for (int rr = 6; rr < rr1 - 6; rr++) {
                for (int cc = 6 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 6; cc += 2, indx += 2) {
                    
                    float Dgrbvvaru = 4.0f * (vcdsq[indx] + vcdsq[indx - v1] + vcdsq[indx - v2] + vcdsq[indx - v3]) - 
                                      SQR(vcd[indx] + vcd[indx - v1] + vcd[indx - v2] + vcd[indx - v3]);
                    float Dgrbvvard = 4.0f * (vcdsq[indx] + vcdsq[indx + v1] + vcdsq[indx + v2] + vcdsq[indx + v3]) - 
                                      SQR(vcd[indx] + vcd[indx + v1] + vcd[indx + v2] + vcd[indx + v3]);
                    float Dgrbhvarl = 4.0f * (hcdsq[indx] + hcdsq[indx - 1] + hcdsq[indx - 2] + hcdsq[indx - 3]) - 
                                      SQR(hcd[indx] + hcd[indx - 1] + hcd[indx - 2] + hcd[indx - 3]);
                    float Dgrbhvarr = 4.0f * (hcdsq[indx] + hcdsq[indx + 1] + hcdsq[indx + 2] + hcdsq[indx + 3]) - 
                                      SQR(hcd[indx] + hcd[indx + 1] + hcd[indx + 2] + hcd[indx + 3]);

                    float hwt = dirwts[indx - 1][1] / (dirwts[indx - 1][1] + dirwts[indx + 1][1]);
                    float vwt = dirwts[indx - v1][0] / (dirwts[indx + v1][0] + dirwts[indx - v1][0]);

                    float vcdvar = epssq + vwt * Dgrbvvard + (1 - vwt) * Dgrbvvaru;
                    float hcdvar = epssq + hwt * Dgrbhvarr + (1 - hwt) * Dgrbhvarl;

                    float Dgrbvvaru1 = dgintv[indx] + dgintv[indx - v1] + dgintv[indx - v2];
                    float Dgrbvvard1 = dgintv[indx] + dgintv[indx + v1] + dgintv[indx + v2];
                    float Dgrbhvarl1 = dginth[indx] + dginth[indx - 1] + dginth[indx - 2];
                    float Dgrbhvarr1 = dginth[indx] + dginth[indx + 1] + dginth[indx + 2];

                    float vcdvar1 = epssq + vwt * Dgrbvvard1 + (1 - vwt) * Dgrbvvaru1;
                    float hcdvar1 = epssq + hwt * Dgrbhvarr1 + (1 - hwt) * Dgrbhvarl1;

                    float varwt = hcdvar / (vcdvar + hcdvar);
                    float diffwt = hcdvar1 / (vcdvar1 + hcdvar1);

                    if ((0.5f - varwt) * (0.5f - diffwt) > 0 && std::abs(0.5f - diffwt) < std::abs(0.5f - varwt)) {
                        hvwt[indx] = varwt;
                    } else {
                        hvwt[indx] = diffwt;
                    }
                }
            }

			// Nyquist test				 
            for (int rr = 6; rr < rr1 - 6; rr++) {
                for (int cc = 6 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 6; cc += 2, indx += 2) {
                    
                    float nyqtest = gaussodd[0] * cddiffsq[indx] +
                                    gaussodd[1] * (cddiffsq[indx - m1] + cddiffsq[indx + p1] + 
                                                   cddiffsq[indx - p1] + cddiffsq[indx + m1]) +
                                    gaussodd[2] * (cddiffsq[indx - v2] + cddiffsq[indx - 2] + 
                                                   cddiffsq[indx + 2] + cddiffsq[indx + v2]) +
                                    gaussodd[3] * (cddiffsq[indx - m2] + cddiffsq[indx + p2] + 
                                                   cddiffsq[indx - p2] + cddiffsq[indx + m2]);

                    nyqtest -= nyqthresh * (gaussgrad[0] * (delhsq[indx] + delvsq[indx]) +
                                            gaussgrad[1] * (delhsq[indx - v1] + delvsq[indx - v1] + 
                                                            delhsq[indx + 1] + delvsq[indx + 1] + 
                                                            delhsq[indx - 1] + delvsq[indx - 1] + 
                                                            delhsq[indx + v1] + delvsq[indx + v1]) +
                                            gaussgrad[2] * (delhsq[indx - m1] + delvsq[indx - m1] + 
                                                            delhsq[indx + p1] + delvsq[indx + p1] + 
                                                            delhsq[indx - p1] + delvsq[indx - p1] + 
                                                            delhsq[indx + m1] + delvsq[indx + m1]) +
                                            gaussgrad[3] * (delhsq[indx - v2] + delvsq[indx - v2] + 
                                                            delhsq[indx - 2] + delvsq[indx - 2] + 
                                                            delhsq[indx + 2] + delvsq[indx + 2] + 
                                                            delhsq[indx + v2] + delvsq[indx + v2]) +
                                            gaussgrad[4] * (delhsq[indx - 2*TS - 1] + delvsq[indx - 2*TS - 1] + 
                                                            delhsq[indx - 2*TS + 1] + delvsq[indx - 2*TS + 1] + 
                                                            delhsq[indx - TS - 2] + delvsq[indx - TS - 2] + 
                                                            delhsq[indx - TS + 2] + delvsq[indx - TS + 2] + 
                                                            delhsq[indx + TS - 2] + delvsq[indx + TS - 2] + 
                                                            delhsq[indx + TS + 2] + delvsq[indx - TS + 2] + 
                                                            delhsq[indx + 2*TS - 1] + delvsq[indx + 2*TS - 1] + 
                                                            delhsq[indx + 2*TS + 1] + delvsq[indx + 2*TS + 1]) +
                                            gaussgrad[5] * (delhsq[indx - m2] + delvsq[indx - m2] + 
                                                            delhsq[indx + p2] + delvsq[indx + p2] + 
                                                            delhsq[indx - p2] + delvsq[indx - p2] + 
                                                            delhsq[indx + m2] + delvsq[indx + m2]));

                    if (nyqtest > 0) nyquist[indx] = 1;
                }
            }

            // Area interpolation for Nyquist regions
            for (int rr = 8; rr < rr1 - 8; rr++) {
                for (int cc = 8 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 8; cc += 2, indx += 2) {

					if (nyquist[indx]) {
                        float sumh = 0, sumv = 0, sumsqh = 0, sumsqv = 0, areawt = 0;
                        for (int i = -6; i < 7; i += 2) {
                            for (int j = -6; j < 7; j += 2) {
                                int indx1 = (rr + i) * TS + cc + j;
								if (nyquist[indx1]) {
                                    sumh += cfa[indx1] - 0.5f * (cfa[indx1 - 1] + cfa[indx1 + 1]);
                                    sumv += cfa[indx1] - 0.5f * (cfa[indx1 - v1] + cfa[indx1 + v1]);
                                    sumsqh += 0.5f * (SQR(cfa[indx1] - cfa[indx1 - 1]) + SQR(cfa[indx1] - cfa[indx1 + 1]));
                                    sumsqv += 0.5f * (SQR(cfa[indx1] - cfa[indx1 - v1]) + SQR(cfa[indx1] - cfa[indx1 + v1]));
                                    areawt += 1;
                                }
                            }
                        }
                        float hcdvar = epssq + std::max(0.0f, areawt * sumsqh - sumh * sumh);
                        float vcdvar = epssq + std::max(0.0f, areawt * sumsqv - sumv * sumv);
                        hvwt[indx] = hcdvar / (vcdvar + hcdvar);
                    }
                }
            }

            // Populate G at R/B sites
            for (int rr = 8; rr < rr1 - 8; rr++) {
                for (int cc = 8 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 8; cc += 2, indx += 2) {
                    
                    float hvwtalt = 0.25f * (hvwt[indx - m1] + hvwt[indx + p1] + hvwt[indx - p1] + hvwt[indx + m1]);
                    float vo = std::abs(0.5f - hvwt[indx]);
                    float ve = std::abs(0.5f - hvwtalt);
                    if (vo < ve) hvwt[indx] = hvwtalt;

                    Dgrb[indx][0] = hcd[indx] * (1 - hvwt[indx]) + vcd[indx] * hvwt[indx];
                    rgb[indx][1] = cfa[indx] + Dgrb[indx][0];

					if (nyquist[indx]) {
                        Dgrbh2[indx] = SQR(rgb[indx][1] - 0.5f * (rgb[indx - 1][1] + rgb[indx + 1][1]));
                        Dgrbv2[indx] = SQR(rgb[indx][1] - 0.5f * (rgb[indx - v1][1] + rgb[indx + v1][1]));
					} else {
						Dgrbh2[indx] = Dgrbv2[indx] = 0;
                    }
                }
            }

            // Refine Nyquist areas
            for (int rr = 8; rr < rr1 - 8; rr++) {
                for (int cc = 8 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 8; cc += 2, indx += 2) {

					if (nyquist[indx]) {
                        float gvarh = epssq + (gquinc[0] * Dgrbh2[indx] +
                                               gquinc[1] * (Dgrbh2[indx - m1] + Dgrbh2[indx + p1] + 
                                                             Dgrbh2[indx - p1] + Dgrbh2[indx + m1]) +
                                               gquinc[2] * (Dgrbh2[indx - v2] + Dgrbh2[indx - 2] + 
                                                             Dgrbh2[indx + 2] + Dgrbh2[indx + v2]) +
                                               gquinc[3] * (Dgrbh2[indx - m2] + Dgrbh2[indx + p2] + 
                                                             Dgrbh2[indx - p2] + Dgrbh2[indx + m2]));
                        float gvarv = epssq + (gquinc[0] * Dgrbv2[indx] +
                                               gquinc[1] * (Dgrbv2[indx - m1] + Dgrbv2[indx + p1] + 
                                                             Dgrbv2[indx - p1] + Dgrbv2[indx + m1]) +
                                               gquinc[2] * (Dgrbv2[indx - v2] + Dgrbv2[indx - 2] + 
                                                             Dgrbv2[indx + 2] + Dgrbv2[indx + v2]) +
                                               gquinc[3] * (Dgrbv2[indx - m2] + Dgrbv2[indx + p2] + 
                                                             Dgrbv2[indx - p2] + Dgrbv2[indx + m2]));
                        Dgrb[indx][0] = (hcd[indx] * gvarv + vcd[indx] * gvarh) / (gvarv + gvarh);
						rgb[indx][1] = cfa[indx] + Dgrb[indx][0];
					}
				}
            }

            // =====================================================
            // 完整的色差插值 (Color Difference Interpolation)
            // =====================================================
            
            // 步骤1: 在 R/B 位置，Dgrb[indx][0] 已经包含了 G-R 或 G-B
            // 现在需要正确分配到 Dgrb[0]=G-R, Dgrb[1]=G-B
            
            // 首先，将 R 位置和 B 位置的色差分别存储
            // R 位置: Dgrb[0] 当前是 G-R，需要计算 G-B
            // B 位置: Dgrb[0] 当前是 G-B，需要计算 G-R，并交换位置
            
            // 步骤1: 先处理 R/B 位置，正确分配 Dgrb[0] 和 Dgrb[1]
            for (int rr = 10; rr < rr1 - 10; rr++) {
                for (int cc = 10 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 10; cc += 2, indx += 2) {
                    
                    int c = FC(rr, cc, start_x, start_y);  // 0=R, 2=B
                    
                    if (c == 0) {
                        // R site: Dgrb[0] = G-R (正确), 需要初始化 Dgrb[1] 为 0
                        // G-R 已经正确存储
                        Dgrb[indx][1] = 0.0f;  // 稍后插值
                    } else if (c == 2) {
                        // B site: Dgrb[0] = G-B, 需要移动到 Dgrb[1]
                        Dgrb[indx][1] = Dgrb[indx][0];  // G-B
                        Dgrb[indx][0] = 0.0f;  // 稍后插值 G-R
                    }
                }
            }
            
            // 步骤2: 在 G 位置插值 G-R 和 G-B
            for (int rr = 11; rr < rr1 - 11; rr++) {
                for (int cc = 11, indx = rr * TS + cc; cc < cc1 - 11; cc++, indx++) {
                    int c = FC(rr, cc, start_x, start_y);
                    
                    if (c == 1) {  // G site
                        // 检查水平邻居是 R 还是 B
                        int c_left = FC(rr, cc - 1, start_x, start_y);
                        
                        if (c_left == 0) {
                            // 水平邻居是 R，垂直邻居是 B
                            // G-R 从水平插值
                            Dgrb[indx][0] = 0.5f * (Dgrb[indx - 1][0] + Dgrb[indx + 1][0]);
                            // G-B 从垂直插值
                            Dgrb[indx][1] = 0.5f * (Dgrb[indx - v1][1] + Dgrb[indx + v1][1]);
                        } else {
                            // 水平邻居是 B，垂直邻居是 R
                            // G-B 从水平插值
                            Dgrb[indx][1] = 0.5f * (Dgrb[indx - 1][1] + Dgrb[indx + 1][1]);
                            // G-R 从垂直插值
                            Dgrb[indx][0] = 0.5f * (Dgrb[indx - v1][0] + Dgrb[indx + v1][0]);
                        }
                    }
                }
            }
            
            // 步骤3: 在 R 位置插值 G-B，在 B 位置插值 G-R (使用对角插值)
            for (int rr = 12; rr < rr1 - 12; rr++) {
                for (int cc = 12 + (FC(rr, 2, start_x, start_y) & 1), indx = rr * TS + cc; 
                     cc < cc1 - 12; cc += 2, indx += 2) {
                    
                    int c = FC(rr, cc, start_x, start_y);  // 0=R, 2=B
                    
                    // 计算对角方向的权重
                    float rbvarp = epssq + (delp[indx] + delp[indx - 1] + delp[indx + 1] + 
                                            delp[indx - v1] + delp[indx + v1]);
                    float rbvarm = epssq + (delm[indx] + delm[indx - 1] + delm[indx + 1] + 
                                            delm[indx - v1] + delm[indx + v1]);
                    float wt = rbvarm / (rbvarp + rbvarm);
                    
                    if (c == 0) {
                        // R site: 需要插值 G-B (Dgrb[1])
                        // 从对角方向的 G 邻居获取 G-B
                        float Dgrb_diag_p = 0.5f * (Dgrb[indx - p1][1] + Dgrb[indx + p1][1]);
                        float Dgrb_diag_m = 0.5f * (Dgrb[indx - m1][1] + Dgrb[indx + m1][1]);
                        Dgrb[indx][1] = wt * Dgrb_diag_p + (1.0f - wt) * Dgrb_diag_m;
                    } else {
                        // B site: 需要插值 G-R (Dgrb[0])
                        // 从对角方向的 G 邻居获取 G-R
                        float Dgrb_diag_p = 0.5f * (Dgrb[indx - p1][0] + Dgrb[indx + p1][0]);
                        float Dgrb_diag_m = 0.5f * (Dgrb[indx - m1][0] + Dgrb[indx + m1][0]);
                        Dgrb[indx][0] = wt * Dgrb_diag_p + (1.0f - wt) * Dgrb_diag_m;
                    }
                }
            }

            // =====================================================
            // 色差域平滑 (Color Difference Domain Smoothing)
            // 使用 3x3 中值滤波抑制孤立伪色点 - 可选，减轻强度
            // =====================================================
            
            // 创建临时缓冲区存储平滑后的色差
            std::vector<float> Dgrb0_smooth(TS * TS);
            std::vector<float> Dgrb1_smooth(TS * TS);
            
            // 应用中值滤波到色差平面
            for (int rr = 13; rr < rr1 - 13; rr++) {
                for (int cc = 13, indx = rr * TS + cc; cc < cc1 - 13; cc++, indx++) {
                    // 使用 3x3 中值滤波
                    float vals0[9], vals1[9];
                    int k = 0;
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int nidx = indx + dr * TS + dc;
                            vals0[k] = Dgrb[nidx][0];
                            vals1[k] = Dgrb[nidx][1];
                            k++;
                        }
                    }
                    std::sort(vals0, vals0 + 9);
                    std::sort(vals1, vals1 + 9);
                    Dgrb0_smooth[indx] = vals0[4];
                    Dgrb1_smooth[indx] = vals1[4];
                }
            }
            
            // 轻度平滑 - 只修正明显的异常值
            for (int rr = 13; rr < rr1 - 13; rr++) {
                for (int cc = 13, indx = rr * TS + cc; cc < cc1 - 13; cc++, indx++) {
                    // 只对偏差很大的点进行平滑
                    float grad0 = std::abs(Dgrb[indx][0] - Dgrb0_smooth[indx]);
                    float grad1 = std::abs(Dgrb[indx][1] - Dgrb1_smooth[indx]);
                    
                    // 提高阈值，只修正真正的异常点
                    const float threshold = 0.1f;  // 10% 的偏差才平滑
                    if (grad0 > threshold) {
                        Dgrb[indx][0] = Dgrb0_smooth[indx];
                    }
                    if (grad1 > threshold) {
                        Dgrb[indx][1] = Dgrb1_smooth[indx];
                    }
                }
            }

            // Final RGB computation
            // Dgrb[0] = G-R, Dgrb[1] = G-B
            // R = G - (G-R) = G - Dgrb[0]
            // B = G - (G-B) = G - Dgrb[1]
            for (int rr = 14; rr < rr1 - 14; rr++) {
                for (int cc = 14, indx = rr * TS + cc; cc < cc1 - 14; cc++, indx++) {
                    rgb[indx][0] = rgb[indx][1] - Dgrb[indx][0];  // R
                    rgb[indx][2] = rgb[indx][1] - Dgrb[indx][1];  // B
                }
            }

            // Copy results back to output
            // 边界需要匹配最后处理阶段的边界 (14)
            for (int rr = 16; rr < rr1 - 16; rr++) {
                int row = rr + top;
                for (int cc = 16; cc < cc1 - 16; cc++) {
                    int col = cc + left;
                    if (row >= 0 && row < height && col >= 0 && col < width) {
                        int indx = row * width + col;
                        int indx1 = rr * TS + cc;
                        red[indx] = rgb[indx1][0] * maxVal;
                        green[indx] = rgb[indx1][1] * maxVal;
                        blue[indx] = rgb[indx1][2] * maxVal;
                    }
                }
            }

            free(buffer);
        }
    }

    // Convert to output Mat
    cv::Mat result(height, width, CV_32FC3);
    for (int y = 0; y < height; ++y) {
        cv::Vec3f* row = result.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            int id = idx(x, y, width);
            row[x][0] = std::clamp(blue[id], 0.0f, maxVal);  // B
            row[x][1] = std::clamp(green[id], 0.0f, maxVal); // G
            row[x][2] = std::clamp(red[id], 0.0f, maxVal);   // R
        }
    }

    result.convertTo(dst, raw.type());
}
