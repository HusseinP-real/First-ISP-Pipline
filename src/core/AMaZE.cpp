#include "AMaZE.h"
#include "demosiac.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// 辅助结构和函数
struct Image {
    int width;
    int height;
    std::vector<float> r, g, b;
    Image(int w, int h) : width(w), height(h), r(w * h), g(w * h), b(w * h) {}
};

enum class CFAColor { Red, GreenR, GreenB, Blue };

inline int idx(int x, int y, int w) { return y * w + x; }

// 根据 Bayer pattern 获取像素颜色类型
inline CFAColor getColor(int x, int y, int start_x, int start_y) {
    bool row_even = ((y + start_y) & 1) == 0;
    bool col_even = ((x + start_x) & 1) == 0;
    if (row_even && col_even) return CFAColor::Red;
    if (row_even && !col_even) return CFAColor::GreenR;
    if (!row_even && col_even) return CFAColor::GreenB;
    return CFAColor::Blue;
}

// 获取 pattern 对应的起始偏移
inline void getPatternOffset(bayerPattern pattern, int& start_x, int& start_y) {
    switch (pattern) {
        case RGGB: start_y = 0; start_x = 0; break;
        case BGGR: start_y = 1; start_x = 1; break;
        case GBRG: start_y = 1; start_x = 0; break;
        case GRBG: start_y = 0; start_x = 1; break;
    }
}

} // namespace

class AMaZEEngine {
public:
    void process(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
        CV_Assert(!raw.empty());
        CV_Assert(raw.channels() == 1);

        const int width = raw.cols;
        const int height = raw.rows;
        CV_Assert(width >= 6 && height >= 6);

        // 转换为 float 便于计算
        cv::Mat raw_f;
        raw.convertTo(raw_f, CV_32F);
        
        // 获取最大值用于 clamp
        double min_raw = 0.0, max_raw = 0.0;
        cv::minMaxLoc(raw_f, &min_raw, &max_raw);
        const float maxVal = static_cast<float>(std::max(1.0, max_raw));

        // 获取 pattern 偏移
        int start_x = 0, start_y = 0;
        getPatternOffset(pattern, start_x, start_y);

        // 边界扩展
        cv::Mat raw_pad;
        cv::copyMakeBorder(raw_f, raw_pad, 3, 3, 3, 3, cv::BORDER_REFLECT_101);

        // Step 1: 自适应方向绿色插值
        std::vector<float> green(width * height);
        interpolateGreen(raw_pad, width, height, start_x, start_y, green);

        // Step 2: 基于色差的红蓝插值
        std::vector<float> red(width * height);
        std::vector<float> blue(width * height);
        interpolateRedBlue(raw_pad, width, height, start_x, start_y, green, red, blue);

        // Step 3: 色彩精炼 (减少拉链效应)
        refineColors(width, height, green, red, blue);

        // Step 4: 转换为输出 Mat
        dst = toMat(width, height, red, green, blue, maxVal, raw.type());
    }

private:
    // 计算单点的水平梯度
    inline float computeGradH(const cv::Mat& raw_pad, int x, int y) {
        auto rp = [&](int yy, int xx) -> float {
            return raw_pad.at<float>(yy + 3, xx + 3);
        };
        float center = rp(y, x);
        return std::abs(rp(y, x - 1) - rp(y, x + 1)) +
               std::abs(2.0f * center - rp(y, x - 2) - rp(y, x + 2));
    }

    // 计算单点的垂直梯度
    inline float computeGradV(const cv::Mat& raw_pad, int x, int y) {
        auto rp = [&](int yy, int xx) -> float {
            return raw_pad.at<float>(yy + 3, xx + 3);
        };
        float center = rp(y, x);
        return std::abs(rp(y - 1, x) - rp(y + 1, x)) +
               std::abs(2.0f * center - rp(y - 2, x) - rp(y + 2, x));
    }

    // 自适应梯度方向的绿色通道插值（使用空间梯度平滑减少迷宫纹）
    void interpolateGreen(const cv::Mat& raw_pad, int w, int h,
                          int start_x, int start_y,
                          std::vector<float>& green) {
        auto rp = [&](int yy, int xx) -> float {
            return raw_pad.at<float>(yy + 3, xx + 3);
        };

        // Step 1: 预计算全图的梯度图
        std::vector<float> gradH_map(w * h, 0.0f);
        std::vector<float> gradV_map(w * h, 0.0f);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int id = idx(x, y, w);
                gradH_map[id] = computeGradH(raw_pad, x, y);
                gradV_map[id] = computeGradV(raw_pad, x, y);
            }
        }

        // Step 2: 进行插值（使用空间积分梯度）
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int id = idx(x, y, w);
                CFAColor c = getColor(x, y, start_x, start_y);

                // 绿色像素直接使用原值
                if (c == CFAColor::GreenR || c == CFAColor::GreenB) {
                    green[id] = rp(y, x);
                    continue;
                }

                // 计算 3x3 邻域内的梯度总和（空间积分梯度，关键改进！）
                float sum_grad_h = 0, sum_grad_v = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = std::clamp(x + dx, 0, w - 1);
                        int ny = std::clamp(y + dy, 0, h - 1);
                        int nid = idx(nx, ny, w);
                        sum_grad_h += gradH_map[nid];
                        sum_grad_v += gradV_map[nid];
                    }
                }

                // Hamilton-Adams 风格的绿色估计
                float center = rp(y, x);
                float est_h = 0.5f * (rp(y, x - 1) + rp(y, x + 1)) +
                              0.25f * (2.0f * center - rp(y, x - 2) - rp(y, x + 2));
                float est_v = 0.5f * (rp(y - 1, x) + rp(y + 1, x)) +
                              0.25f * (2.0f * center - rp(y - 2, x) - rp(y + 2, x));

                // 稳健的权重分配（引入不确定区阈值）
                float diff = std::abs(sum_grad_h - sum_grad_v);
                float total = sum_grad_h + sum_grad_v + 1e-4f;

                if (diff / total < 0.15f) {
                    // 梯度非常接近的区域：强制平均，防止迷宫纹
                    green[id] = std::max(0.0f, 0.5f * (est_h + est_v));
                } else {
                    // 边缘明确的区域：使用梯度平方倒数加权
                    const float eps = 1e-4f;
                    float wh = 1.0f / (sum_grad_h * sum_grad_h + eps);
                    float wv = 1.0f / (sum_grad_v * sum_grad_v + eps);
                    green[id] = std::max(0.0f, (wh * est_h + wv * est_v) / (wh + wv));
                }
            }
        }
    }

    // 基于色彩差分的红蓝通道插值
    void interpolateRedBlue(const cv::Mat& raw_pad, int w, int h,
                            int start_x, int start_y,
                            const std::vector<float>& green,
                            std::vector<float>& red,
                            std::vector<float>& blue) {
        auto rp = [&](int yy, int xx) -> float {
            return raw_pad.at<float>(yy + 3, xx + 3);
        };

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int id = idx(x, y, w);
                CFAColor c = getColor(x, y, start_x, start_y);

                switch (c) {
                    case CFAColor::Red: {
                        // 红色位置：红色已知，蓝色通过对角邻居插值（考虑边缘方向）
                        red[id] = rp(y, x);
                        
                        // 计算两个对角线方向的梯度
                        float grad_d1 = std::abs(rp(y - 1, x - 1) - rp(y + 1, x + 1)); // NW-SE
                        float grad_d2 = std::abs(rp(y - 1, x + 1) - rp(y + 1, x - 1)); // NE-SW
                        
                        const float eps = 1.0f;
                        float w1 = 1.0f / (grad_d1 + eps); // NW-SE 方向权重
                        float w2 = 1.0f / (grad_d2 + eps); // NE-SW 方向权重
                        
                        // 按梯度加权平均对角邻居的色差
                        float diff1 = 0, diff2 = 0;
                        int cnt1 = 0, cnt2 = 0;
                        
                        // NW-SE 对角线
                        if (x > 0 && y > 0) {
                            diff1 += rp(y - 1, x - 1) - green[idx(x - 1, y - 1, w)];
                            cnt1++;
                        }
                        if (x < w - 1 && y < h - 1) {
                            diff1 += rp(y + 1, x + 1) - green[idx(x + 1, y + 1, w)];
                            cnt1++;
                        }
                        
                        // NE-SW 对角线
                        if (x < w - 1 && y > 0) {
                            diff2 += rp(y - 1, x + 1) - green[idx(x + 1, y - 1, w)];
                            cnt2++;
                        }
                        if (x > 0 && y < h - 1) {
                            diff2 += rp(y + 1, x - 1) - green[idx(x - 1, y + 1, w)];
                            cnt2++;
                        }
                        
                        float avg1 = (cnt1 > 0) ? diff1 / cnt1 : 0;
                        float avg2 = (cnt2 > 0) ? diff2 / cnt2 : 0;
                        
                        blue[id] = green[id] + (w1 * avg1 + w2 * avg2) / (w1 + w2);
                        break;
                    }
                    case CFAColor::Blue: {
                        // 蓝色位置：蓝色已知，红色通过对角邻居插值（考虑边缘方向）
                        blue[id] = rp(y, x);
                        
                        float grad_d1 = std::abs(rp(y - 1, x - 1) - rp(y + 1, x + 1));
                        float grad_d2 = std::abs(rp(y - 1, x + 1) - rp(y + 1, x - 1));
                        
                        const float eps = 1.0f;
                        float w1 = 1.0f / (grad_d1 + eps);
                        float w2 = 1.0f / (grad_d2 + eps);
                        
                        float diff1 = 0, diff2 = 0;
                        int cnt1 = 0, cnt2 = 0;
                        
                        if (x > 0 && y > 0) {
                            diff1 += rp(y - 1, x - 1) - green[idx(x - 1, y - 1, w)];
                            cnt1++;
                        }
                        if (x < w - 1 && y < h - 1) {
                            diff1 += rp(y + 1, x + 1) - green[idx(x + 1, y + 1, w)];
                            cnt1++;
                        }
                        
                        if (x < w - 1 && y > 0) {
                            diff2 += rp(y - 1, x + 1) - green[idx(x + 1, y - 1, w)];
                            cnt2++;
                        }
                        if (x > 0 && y < h - 1) {
                            diff2 += rp(y + 1, x - 1) - green[idx(x - 1, y + 1, w)];
                            cnt2++;
                        }
                        
                        float avg1 = (cnt1 > 0) ? diff1 / cnt1 : 0;
                        float avg2 = (cnt2 > 0) ? diff2 / cnt2 : 0;
                        
                        red[id] = green[id] + (w1 * avg1 + w2 * avg2) / (w1 + w2);
                        break;
                    }
                    case CFAColor::GreenR: {
                        // 红行绿色位置：红色在左右，蓝色在上下
                        float diff_r = 0, diff_b = 0;
                        int cnt_r = 0, cnt_b = 0;
                        
                        // 红色：水平邻居
                        if (x > 0) {
                            diff_r += rp(y, x - 1) - green[idx(x - 1, y, w)];
                            cnt_r++;
                        }
                        if (x < w - 1) {
                            diff_r += rp(y, x + 1) - green[idx(x + 1, y, w)];
                            cnt_r++;
                        }
                        // 蓝色：垂直邻居
                        if (y > 0) {
                            diff_b += rp(y - 1, x) - green[idx(x, y - 1, w)];
                            cnt_b++;
                        }
                        if (y < h - 1) {
                            diff_b += rp(y + 1, x) - green[idx(x, y + 1, w)];
                            cnt_b++;
                        }
                        
                        red[id] = green[id] + (cnt_r > 0 ? diff_r / cnt_r : 0);
                        blue[id] = green[id] + (cnt_b > 0 ? diff_b / cnt_b : 0);
                        break;
                    }
                    case CFAColor::GreenB: {
                        // 蓝行绿色位置：蓝色在左右，红色在上下
                        float diff_r = 0, diff_b = 0;
                        int cnt_r = 0, cnt_b = 0;
                        
                        // 蓝色：水平邻居
                        if (x > 0) {
                            diff_b += rp(y, x - 1) - green[idx(x - 1, y, w)];
                            cnt_b++;
                        }
                        if (x < w - 1) {
                            diff_b += rp(y, x + 1) - green[idx(x + 1, y, w)];
                            cnt_b++;
                        }
                        // 红色：垂直邻居
                        if (y > 0) {
                            diff_r += rp(y - 1, x) - green[idx(x, y - 1, w)];
                            cnt_r++;
                        }
                        if (y < h - 1) {
                            diff_r += rp(y + 1, x) - green[idx(x, y + 1, w)];
                            cnt_r++;
                        }
                        
                        red[id] = green[id] + (cnt_r > 0 ? diff_r / cnt_r : 0);
                        blue[id] = green[id] + (cnt_b > 0 ? diff_b / cnt_b : 0);
                        break;
                    }
                }
            }
        }
    }

    // 色彩精炼：使用 3x3 中值滤波平滑色差（简化版，避免过度平滑）
    void refineColors(int w, int h,
                      std::vector<float>& green,
                      std::vector<float>& red,
                      std::vector<float>& blue) {
        // 计算色差
        std::vector<float> rg(w * h);
        std::vector<float> bg(w * h);

        for (int i = 0; i < w * h; ++i) {
            rg[i] = red[i] - green[i];
            bg[i] = blue[i] - green[i];
        }

        cv::Mat rgMat(h, w, CV_32F, rg.data());
        cv::Mat bgMat(h, w, CV_32F, bg.data());

        // 使用 3x3 中值滤波平滑色差（足够去除离群点，又不会过度平滑）
        cv::Mat rgMed, bgMed;
        cv::medianBlur(rgMat, rgMed, 3);
        cv::medianBlur(bgMat, bgMed, 3);

        // 重建红蓝通道
        for (int y = 0; y < h; ++y) {
            const float* rgRow = rgMed.ptr<float>(y);
            const float* bgRow = bgMed.ptr<float>(y);
            for (int x = 0; x < w; ++x) {
                int id = idx(x, y, w);
                red[id] = green[id] + rgRow[x];
                blue[id] = green[id] + bgRow[x];
            }
        }
    }

    // 转换为输出 Mat
    cv::Mat toMat(int w, int h,
                  const std::vector<float>& red,
                  const std::vector<float>& green,
                  const std::vector<float>& blue,
                  float maxVal, int raw_type) {
        cv::Mat f(h, w, CV_32FC3);
        for (int y = 0; y < h; ++y) {
            cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < w; ++x) {
                int id = idx(x, y, w);
                // BGR 顺序
                row[x][0] = std::clamp(blue[id], 0.0f, maxVal);
                row[x][1] = std::clamp(green[id], 0.0f, maxVal);
                row[x][2] = std::clamp(red[id], 0.0f, maxVal);
            }
        }

        cv::Mat dst;
        f.convertTo(dst, CV_MAKETYPE(CV_MAT_DEPTH(raw_type), 3));
        return dst;
    }
};

void demosiacAMaZE(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    AMaZEEngine engine;
    engine.process(raw, dst, pattern);
}
