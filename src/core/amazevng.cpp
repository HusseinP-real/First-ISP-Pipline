#include "amazevng.h"
#include "demosiac.h"
#include "vng.h"
#include "AMaZE.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace {

enum class CFAColor { Red, GreenR, GreenB, Blue };

inline int idx(int x, int y, int w) { return y * w + x; }

inline CFAColor getColor(int x, int y, int start_x, int start_y) {
    bool row_even = ((y + start_y) & 1) == 0;
    bool col_even = ((x + start_x) & 1) == 0;
    if (row_even && col_even) return CFAColor::Red;
    if (row_even && !col_even) return CFAColor::GreenR;
    if (!row_even && col_even) return CFAColor::GreenB;
    return CFAColor::Blue;
}

inline void getPatternOffset(bayerPattern pattern, int& start_x, int& start_y) {
    switch (pattern) {
        case RGGB: start_y = 0; start_x = 0; break;
        case BGGR: start_y = 1; start_x = 1; break;
        case GBRG: start_y = 1; start_x = 0; break;
        case GRBG: start_y = 0; start_x = 1; break;
    }
}

// 计算单点的水平梯度
inline float computeGradH(const cv::Mat& raw_f, int x, int y) {
    const int w = raw_f.cols;
    const int h = raw_f.rows;
    
    auto get = [&](int yy, int xx) -> float {
        xx = std::clamp(xx, 0, w - 1);
        yy = std::clamp(yy, 0, h - 1);
        return raw_f.at<float>(yy, xx);
    };
    
    float center = get(y, x);
    return std::abs(get(y, x - 1) - get(y, x + 1)) +
           std::abs(2.0f * center - get(y, x - 2) - get(y, x + 2));
}

// 计算单点的垂直梯度
inline float computeGradV(const cv::Mat& raw_f, int x, int y) {
    const int w = raw_f.cols;
    const int h = raw_f.rows;
    
    auto get = [&](int yy, int xx) -> float {
        xx = std::clamp(xx, 0, w - 1);
        yy = std::clamp(yy, 0, h - 1);
        return raw_f.at<float>(yy, xx);
    };
    
    float center = get(y, x);
    return std::abs(get(y - 1, x) - get(y + 1, x)) +
           std::abs(2.0f * center - get(y - 2, x) - get(y + 2, x));
}

} // namespace

void demosiacAMaZEVNG(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    CV_Assert(!raw.empty());
    CV_Assert(raw.channels() == 1);

    const int width = raw.cols;
    const int height = raw.rows;

    int start_x = 0, start_y = 0;
    getPatternOffset(pattern, start_x, start_y);

    // Step 1: 调用现有的 VNG 算法（获取完整结果，主要用绿色通道）
    cv::Mat vng_result;
    demosiacVNG(raw, vng_result, pattern);
    
    cv::Mat vng_f;
    vng_result.convertTo(vng_f, CV_32FC3);

    // Step 2: 计算梯度不确定性
    cv::Mat raw_f;
    raw.convertTo(raw_f, CV_32F);

    cv::Mat gradH_map(height, width, CV_32F);
    cv::Mat gradV_map(height, width, CV_32F);

    for (int y = 0; y < height; ++y) {
        float* gradH_row = gradH_map.ptr<float>(y);
        float* gradV_row = gradV_map.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            gradH_row[x] = computeGradH(raw_f, x, y);
            gradV_row[x] = computeGradV(raw_f, x, y);
        }
    }

    // 5x5 空间积分
    cv::Mat gradH_sum, gradV_sum;
    cv::boxFilter(gradH_map, gradH_sum, CV_32F, cv::Size(5, 5), cv::Point(-1, -1), true, cv::BORDER_REFLECT_101);
    cv::boxFilter(gradV_map, gradV_sum, CV_32F, cv::Size(5, 5), cv::Point(-1, -1), true, cv::BORDER_REFLECT_101);

    cv::Mat uncertainty(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
        const float* sumH = gradH_sum.ptr<float>(y);
        const float* sumV = gradV_sum.ptr<float>(y);
        float* unc = uncertainty.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            float diff = std::abs(sumH[x] - sumV[x]);
            float total = sumH[x] + sumV[x] + 1e-4f;
            unc[x] = diff / total;
        }
    }

    // 诊断
    double min_unc, max_unc, mean_unc;
    cv::minMaxLoc(uncertainty, &min_unc, &max_unc);
    mean_unc = cv::mean(uncertainty)[0];
    std::cout << "[AMaZE-VNG] Gradient uncertainty: min=" << min_unc 
              << " max=" << max_unc << " mean=" << mean_unc << std::endl;

    // Step 3: 构建混合绿色通道
    // 在梯度不确定区域使用 VNG 的绿色（多方向平均，抗迷宫纹）
    // 在边缘明确区域使用 AMaZE 风格的梯度自适应绿色
    
    cv::Mat raw_pad;
    cv::copyMakeBorder(raw_f, raw_pad, 3, 3, 3, 3, cv::BORDER_REFLECT_101);
    
    auto rp = [&](int yy, int xx) -> float {
        return raw_pad.at<float>(yy + 3, xx + 3);
    };

    std::vector<float> green(width * height);
    int count_vng = 0, count_blend = 0, count_amaze = 0;

    // 更激进的阈值：让更多像素使用 VNG
    const float thresh_low = 0.20f;   // 低于此值：完全使用 VNG
    const float thresh_high = 0.50f;  // 高于此值：完全使用 AMaZE

    for (int y = 0; y < height; ++y) {
        const float* unc_row = uncertainty.ptr<float>(y);
        const cv::Vec3f* vng_row = vng_f.ptr<cv::Vec3f>(y);
        
        for (int x = 0; x < width; ++x) {
            int id = idx(x, y, width);
            CFAColor c = getColor(x, y, start_x, start_y);

            // 绿色像素直接使用原值
            if (c == CFAColor::GreenR || c == CFAColor::GreenB) {
                green[id] = rp(y, x);
                continue;
            }

            // VNG 的绿色估计
            float green_vng = vng_row[x][1];

            // AMaZE 风格的绿色估计（Hamilton-Adams + 梯度加权）
            float center = rp(y, x);
            float est_h = 0.5f * (rp(y, x - 1) + rp(y, x + 1)) +
                          0.25f * (2.0f * center - rp(y, x - 2) - rp(y, x + 2));
            float est_v = 0.5f * (rp(y - 1, x) + rp(y + 1, x)) +
                          0.25f * (2.0f * center - rp(y - 2, x) - rp(y + 2, x));

            float grad_h = computeGradH(raw_f, x, y);
            float grad_v = computeGradV(raw_f, x, y);
            const float eps = 1e-4f;
            float wh = 1.0f / (grad_h * grad_h + eps);
            float wv = 1.0f / (grad_v * grad_v + eps);
            float green_amaze = (wh * est_h + wv * est_v) / (wh + wv);

            // 根据不确定性混合
            float ratio = unc_row[x];
            
            if (ratio < thresh_low) {
                green[id] = green_vng;
                count_vng++;
            } else if (ratio > thresh_high) {
                green[id] = green_amaze;
                count_amaze++;
            } else {
                float t = (ratio - thresh_low) / (thresh_high - thresh_low);
                green[id] = (1.0f - t) * green_vng + t * green_amaze;
                count_blend++;
            }
            
            green[id] = std::max(0.0f, green[id]);
        }
    }

    // 诊断输出（只统计非绿色像素）
    int non_green_total = count_vng + count_blend + count_amaze;
    if (non_green_total > 0) {
        std::cout << "[AMaZE-VNG] Green channel: VNG=" << 100.0*count_vng/non_green_total << "%, "
                  << "Blend=" << 100.0*count_blend/non_green_total << "%, "
                  << "AMaZE=" << 100.0*count_amaze/non_green_total << "%" << std::endl;
    }

    // Step 4: 使用混合绿色通道 + AMaZE 风格的色差红蓝插值
    std::vector<float> red(width * height);
    std::vector<float> blue(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int id = idx(x, y, width);
            CFAColor c = getColor(x, y, start_x, start_y);

            switch (c) {
                case CFAColor::Red: {
                    red[id] = rp(y, x);
                    
                    float grad_d1 = std::abs(rp(y - 1, x - 1) - rp(y + 1, x + 1));
                    float grad_d2 = std::abs(rp(y - 1, x + 1) - rp(y + 1, x - 1));
                    
                    const float eps = 1.0f;
                    float w1 = 1.0f / (grad_d1 + eps);
                    float w2 = 1.0f / (grad_d2 + eps);
                    
                    float diff1 = 0, diff2 = 0;
                    int cnt1 = 0, cnt2 = 0;
                    
                    if (x > 0 && y > 0) {
                        diff1 += rp(y - 1, x - 1) - green[idx(x - 1, y - 1, width)];
                        cnt1++;
                    }
                    if (x < width - 1 && y < height - 1) {
                        diff1 += rp(y + 1, x + 1) - green[idx(x + 1, y + 1, width)];
                        cnt1++;
                    }
                    
                    if (x < width - 1 && y > 0) {
                        diff2 += rp(y - 1, x + 1) - green[idx(x + 1, y - 1, width)];
                        cnt2++;
                    }
                    if (x > 0 && y < height - 1) {
                        diff2 += rp(y + 1, x - 1) - green[idx(x - 1, y + 1, width)];
                        cnt2++;
                    }
                    
                    float avg1 = (cnt1 > 0) ? diff1 / cnt1 : 0;
                    float avg2 = (cnt2 > 0) ? diff2 / cnt2 : 0;
                    
                    blue[id] = green[id] + (w1 * avg1 + w2 * avg2) / (w1 + w2);
                    break;
                }
                case CFAColor::Blue: {
                    blue[id] = rp(y, x);
                    
                    float grad_d1 = std::abs(rp(y - 1, x - 1) - rp(y + 1, x + 1));
                    float grad_d2 = std::abs(rp(y - 1, x + 1) - rp(y + 1, x - 1));
                    
                    const float eps = 1.0f;
                    float w1 = 1.0f / (grad_d1 + eps);
                    float w2 = 1.0f / (grad_d2 + eps);
                    
                    float diff1 = 0, diff2 = 0;
                    int cnt1 = 0, cnt2 = 0;
                    
                    if (x > 0 && y > 0) {
                        diff1 += rp(y - 1, x - 1) - green[idx(x - 1, y - 1, width)];
                        cnt1++;
                    }
                    if (x < width - 1 && y < height - 1) {
                        diff1 += rp(y + 1, x + 1) - green[idx(x + 1, y + 1, width)];
                        cnt1++;
                    }
                    
                    if (x < width - 1 && y > 0) {
                        diff2 += rp(y - 1, x + 1) - green[idx(x + 1, y - 1, width)];
                        cnt2++;
                    }
                    if (x > 0 && y < height - 1) {
                        diff2 += rp(y + 1, x - 1) - green[idx(x - 1, y + 1, width)];
                        cnt2++;
                    }
                    
                    float avg1 = (cnt1 > 0) ? diff1 / cnt1 : 0;
                    float avg2 = (cnt2 > 0) ? diff2 / cnt2 : 0;
                    
                    red[id] = green[id] + (w1 * avg1 + w2 * avg2) / (w1 + w2);
                    break;
                }
                case CFAColor::GreenR: {
                    float diff_r = 0, diff_b = 0;
                    int cnt_r = 0, cnt_b = 0;
                    
                    if (x > 0) {
                        diff_r += rp(y, x - 1) - green[idx(x - 1, y, width)];
                        cnt_r++;
                    }
                    if (x < width - 1) {
                        diff_r += rp(y, x + 1) - green[idx(x + 1, y, width)];
                        cnt_r++;
                    }
                    if (y > 0) {
                        diff_b += rp(y - 1, x) - green[idx(x, y - 1, width)];
                        cnt_b++;
                    }
                    if (y < height - 1) {
                        diff_b += rp(y + 1, x) - green[idx(x, y + 1, width)];
                        cnt_b++;
                    }
                    
                    red[id] = green[id] + (cnt_r > 0 ? diff_r / cnt_r : 0);
                    blue[id] = green[id] + (cnt_b > 0 ? diff_b / cnt_b : 0);
                    break;
                }
                case CFAColor::GreenB: {
                    float diff_r = 0, diff_b = 0;
                    int cnt_r = 0, cnt_b = 0;
                    
                    if (x > 0) {
                        diff_b += rp(y, x - 1) - green[idx(x - 1, y, width)];
                        cnt_b++;
                    }
                    if (x < width - 1) {
                        diff_b += rp(y, x + 1) - green[idx(x + 1, y, width)];
                        cnt_b++;
                    }
                    if (y > 0) {
                        diff_r += rp(y - 1, x) - green[idx(x, y - 1, width)];
                        cnt_r++;
                    }
                    if (y < height - 1) {
                        diff_r += rp(y + 1, x) - green[idx(x, y + 1, width)];
                        cnt_r++;
                    }
                    
                    red[id] = green[id] + (cnt_r > 0 ? diff_r / cnt_r : 0);
                    blue[id] = green[id] + (cnt_b > 0 ? diff_b / cnt_b : 0);
                    break;
                }
            }
        }
    }

    // Step 5: 色差精炼（中值滤波）
    std::vector<float> rg(width * height);
    std::vector<float> bg(width * height);

    for (int i = 0; i < width * height; ++i) {
        rg[i] = red[i] - green[i];
        bg[i] = blue[i] - green[i];
    }

    cv::Mat rgMat(height, width, CV_32F, rg.data());
    cv::Mat bgMat(height, width, CV_32F, bg.data());

    cv::Mat rgMed, bgMed;
    cv::medianBlur(rgMat, rgMed, 3);
    cv::medianBlur(bgMat, bgMed, 3);

    for (int y = 0; y < height; ++y) {
        const float* rgRow = rgMed.ptr<float>(y);
        const float* bgRow = bgMed.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            int id = idx(x, y, width);
            red[id] = green[id] + rgRow[x];
            blue[id] = green[id] + bgRow[x];
        }
    }

    // Step 6: 输出
    double min_raw, max_raw;
    cv::minMaxLoc(raw_f, &min_raw, &max_raw);
    const float maxVal = static_cast<float>(std::max(1.0, max_raw));

    cv::Mat f(height, width, CV_32FC3);
    for (int y = 0; y < height; ++y) {
        cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            int id = idx(x, y, width);
            row[x][0] = std::clamp(blue[id], 0.0f, maxVal);
            row[x][1] = std::clamp(green[id], 0.0f, maxVal);
            row[x][2] = std::clamp(red[id], 0.0f, maxVal);
        }
    }

    f.convertTo(dst, CV_MAKETYPE(raw.depth(), 3));
}
