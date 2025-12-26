#include "ahd.h"
#include "demosiac.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// 简单的图像数据结构（存 float 便于计算）
struct Image {
    int width;
    int height;
    std::vector<float> r, g, b;
    Image(int w, int h) : width(w), height(h), r(w * h), g(w * h), b(w * h) {}
};

// 存储中间方向结果的结构
struct DirectionalResult {
    Image img;
    std::vector<int> homogeneity; // 同质性得分
    DirectionalResult(int w, int h) : img(w, h), homogeneity(w * h, 0) {}
};

enum class CFAColor { Red, GreenR, GreenB, Blue };

inline int idx(int x, int y, int w) { return y * w + x; }

// 根据 Bayer 奇偶判定像素颜色（以 BGGR 为参考，start_x/start_y 用来平移模式）
inline CFAColor getColor(int x, int y, int start_x, int start_y) {
    // 以 RGGB 为基准： (even, even)=R, (even, odd)=G(R行), (odd, even)=G(B行), (odd, odd)=B
    bool row_even = ((y + start_y) & 1) == 0;
    bool col_even = ((x + start_x) & 1) == 0;
    if (row_even && col_even) return CFAColor::Red;
    if (row_even && !col_even) return CFAColor::GreenR;
    if (!row_even && col_even) return CFAColor::GreenB;
    return CFAColor::Blue;
}

} // namespace

class AHDEngine {
public:
    void process(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
        CV_Assert(!raw.empty());
        CV_Assert(raw.channels() == 1);
        CV_Assert(pattern == BGGR && "Current AHD implementation validated for BGGR.");

        const int width = raw.cols;
        const int height = raw.rows;
        CV_Assert(width >= 6 && height >= 6); // 需要 5x5 邻域

        // 1) 基线 HA 结果，用于边界和初始化
        cv::Mat base16;
        demosiac(raw, base16, pattern); // CV_8/16UC3

        // 2) 转 float 便于计算，同时填充 DirectionalResult 的初值（边界直接用基线）
        DirectionalResult horiz(width, height);
        DirectionalResult vert(width, height);
        fillFromBase(base16, horiz.img);
        fillFromBase(base16, vert.img);

        cv::Mat raw_f;
        raw.convertTo(raw_f, CV_32F);
        double min_raw = 0.0, max_raw = 0.0;
        cv::minMaxLoc(raw_f, &min_raw, &max_raw);
        const float maxVal = static_cast<float>(std::max(1.0, max_raw)); // 避免 0
        const float epsilon = maxVal * 0.04f; // 位深自适应阈值
        const float epsSq = epsilon * epsilon;
        cv::Mat raw_pad;
        cv::copyMakeBorder(raw_f, raw_pad, 2, 2, 2, 2, cv::BORDER_REFLECT_101);

        int start_y = 0, start_x = 0;
        switch (pattern) {
            case RGGB: start_y = 0; start_x = 0; break;
            case BGGR: start_y = 1; start_x = 1; break;
            case GBRG: start_y = 1; start_x = 0; break;
            case GRBG: start_y = 0; start_x = 1; break;
        }

        // 3) 预计算两个方向的 G 平面
        std::vector<float> Gh(width * height, 0.0f);
        std::vector<float> Gv(width * height, 0.0f);
        computeGreen(raw_pad, width, height, start_x, start_y, Gh, Gv);

        // 4) 方向性插值 RGB
        interpolate_directionally(raw_pad, width, height, start_x, start_y, Gh, Gv, horiz, vert);

        // 5) 同质性评估
        calculate_homogeneity(horiz, epsSq);
        calculate_homogeneity(vert, epsSq);

        // 6) 选择更平滑的方向
        Image final_img(width, height);
        combine_results(horiz, vert, final_img);

        // 7) 伪彩去除：对 (R-G) / (B-G) 做中值滤波
        refine_color_artifacts(final_img, maxVal);

        // 8) 转回与输入一致的位深
        dst = toMat(final_img, raw.type());
    }

private:
    // 将基线 demosaic 的结果填入 Image，作为边界/默认值
    void fillFromBase(const cv::Mat& base, Image& img) {
        cv::Mat base_f;
        base.convertTo(base_f, CV_32F);
        for (int y = 0; y < img.height; ++y) {
            const cv::Vec3f* row = base_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < img.width; ++x) {
                int id = idx(x, y, img.width);
                img.b[id] = row[x][0];
                img.g[id] = row[x][1];
                img.r[id] = row[x][2];
            }
        }
    }

    // 计算水平 / 垂直方向的 G 候选（Hamilton-Adams 形式）
    void computeGreen(const cv::Mat& raw_pad,
                      int w, int h,
                      int start_x, int start_y,
                      std::vector<float>& Gh,
                      std::vector<float>& Gv) {
        auto rp = [&](int yy, int xx) -> float {
            return raw_pad.at<float>(yy + 2, xx + 2);
        };

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int id = idx(x, y, w);
                CFAColor c = getColor(x, y, start_x, start_y);
                float center = rp(y, x);

                if (c == CFAColor::GreenB || c == CFAColor::GreenR) {
                    Gh[id] = center;
                    Gv[id] = center;
                    continue;
                }

                // Hamilton-Adams 梯度引导插值
                float gh = 0.5f * (rp(y, x - 1) + rp(y, x + 1)) +
                           0.25f * (2 * center - rp(y, x - 2) - rp(y, x + 2));
                float gv = 0.5f * (rp(y - 1, x) + rp(y + 1, x)) +
                           0.25f * (2 * center - rp(y - 2, x) - rp(y + 2, x));

                Gh[id] = std::max(0.0f, gh);
                Gv[id] = std::max(0.0f, gv);
            }
        }
    }

    // 方向性插值：利用预计算的 G 候选生成完整 RGB 候选图
    void interpolate_directionally(const cv::Mat& raw_pad,
                                   int w, int h,
                                   int start_x, int start_y,
                                   const std::vector<float>& Gh,
                                   const std::vector<float>& Gv,
                                   DirectionalResult& horiz,
                                   DirectionalResult& vert) {
        auto rp = [&](int yy, int xx) -> float {
            return raw_pad.at<float>(yy + 2, xx + 2);
        };

        for (int y = 2; y < h - 2; ++y) {
            for (int x = 2; x < w - 2; ++x) {
                int id = idx(x, y, w);
                CFAColor c = getColor(x, y, start_x, start_y);

                const std::vector<float>& G_h = Gh;
                const std::vector<float>& G_v = Gv;

                // ------ 水平方向候选 ------
                float gH = G_h[id];
                float rH = 0, bH = 0;

                switch (c) {
                    case CFAColor::Red:
                        rH = rp(y, x);
                        bH = gH + ((rp(y - 1, x - 1) - G_h[idx(x - 1, y - 1, w)]) +
                                   (rp(y - 1, x + 1) - G_h[idx(x + 1, y - 1, w)]) +
                                   (rp(y + 1, x - 1) - G_h[idx(x - 1, y + 1, w)]) +
                                   (rp(y + 1, x + 1) - G_h[idx(x + 1, y + 1, w)])) * 0.25f;
                        break;
                    case CFAColor::Blue:
                        bH = rp(y, x);
                        rH = gH + ((rp(y - 1, x - 1) - G_h[idx(x - 1, y - 1, w)]) +
                                   (rp(y - 1, x + 1) - G_h[idx(x + 1, y - 1, w)]) +
                                   (rp(y + 1, x - 1) - G_h[idx(x - 1, y + 1, w)]) +
                                   (rp(y + 1, x + 1) - G_h[idx(x + 1, y + 1, w)])) * 0.25f;
                        break;
                    case CFAColor::GreenB: {
                        // Green on Blue row: R from vertical (R 上下), B from horizontal (B 左右)
                        bH = gH + 0.5f * ((rp(y, x - 1) - G_h[idx(x - 1, y, w)]) +
                                          (rp(y, x + 1) - G_h[idx(x + 1, y, w)]));
                        rH = gH + 0.5f * ((rp(y - 1, x) - G_h[idx(x, y - 1, w)]) +
                                          (rp(y + 1, x) - G_h[idx(x, y + 1, w)]));
                        break;
                    }
                    case CFAColor::GreenR: {
                        // Green on Red row: R from horizontal, B from vertical
                        rH = gH + 0.5f * ((rp(y, x - 1) - G_h[idx(x - 1, y, w)]) +
                                          (rp(y, x + 1) - G_h[idx(x + 1, y, w)]));
                        bH = gH + 0.5f * ((rp(y - 1, x) - G_h[idx(x, y - 1, w)]) +
                                          (rp(y + 1, x) - G_h[idx(x, y + 1, w)]));
                        break;
                    }
                }

                horiz.img.r[id] = rH;
                horiz.img.g[id] = gH;
                horiz.img.b[id] = bH;

                // ------ 垂直方向候选 ------
                float gV = G_v[id];
                float rV = 0, bV = 0;

                switch (c) {
                    case CFAColor::Red:
                        rV = rp(y, x);
                        bV = gV + ((rp(y - 1, x - 1) - G_v[idx(x - 1, y - 1, w)]) +
                                   (rp(y - 1, x + 1) - G_v[idx(x + 1, y - 1, w)]) +
                                   (rp(y + 1, x - 1) - G_v[idx(x - 1, y + 1, w)]) +
                                   (rp(y + 1, x + 1) - G_v[idx(x + 1, y + 1, w)])) * 0.25f;
                        break;
                    case CFAColor::Blue:
                        bV = rp(y, x);
                        rV = gV + ((rp(y - 1, x - 1) - G_v[idx(x - 1, y - 1, w)]) +
                                   (rp(y - 1, x + 1) - G_v[idx(x + 1, y - 1, w)]) +
                                   (rp(y + 1, x - 1) - G_v[idx(x - 1, y + 1, w)]) +
                                   (rp(y + 1, x + 1) - G_v[idx(x + 1, y + 1, w)])) * 0.25f;
                        break;
                    case CFAColor::GreenB: {
                        bV = gV + 0.5f * ((rp(y, x - 1) - G_v[idx(x - 1, y, w)]) +
                                          (rp(y, x + 1) - G_v[idx(x + 1, y, w)]));
                        rV = gV + 0.5f * ((rp(y - 1, x) - G_v[idx(x, y - 1, w)]) +
                                          (rp(y + 1, x) - G_v[idx(x, y + 1, w)]));
                        break;
                    }
                    case CFAColor::GreenR: {
                        rV = gV + 0.5f * ((rp(y, x - 1) - G_v[idx(x - 1, y, w)]) +
                                          (rp(y, x + 1) - G_v[idx(x + 1, y, w)]));
                        bV = gV + 0.5f * ((rp(y - 1, x) - G_v[idx(x, y - 1, w)]) +
                                          (rp(y + 1, x) - G_v[idx(x, y + 1, w)]));
                        break;
                    }
                }

                vert.img.r[id] = rV;
                vert.img.g[id] = gV;
                vert.img.b[id] = bV;
            }
        }
    }

    // 计算同质性得分 (AHD 的核心)
    void calculate_homogeneity(DirectionalResult& res, float epsSq) {
        const int w = res.img.width;
        const int h = res.img.height;

        cv::parallel_for_(cv::Range(2, h - 2), [&](const cv::Range& r) {
            for (int y = r.start; y < r.end; ++y) {
                for (int x = 2; x < w - 2; ++x) {
                    int score = 0;
                    int cid = idx(x, y, w);
                    float cr = res.img.r[cid];
                    float cg = res.img.g[cid];
                    float cb = res.img.b[cid];

                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            int nid = idx(x + dx, y + dy, w);
                            float dr = cr - res.img.r[nid];
                            float dg = cg - res.img.g[nid];
                            float db = cb - res.img.b[nid];
                            float distSq = dr * dr + dg * dg + db * db;
                            if (distSq < epsSq) score++;
                        }
                    }
                    res.homogeneity[cid] = score;
                }
            }
        });
    }

    // 根据同质性得分融合结果
    void combine_results(const DirectionalResult& horiz, const DirectionalResult& vert, Image& out) {
        const int total = out.width * out.height;
        for (int i = 0; i < total; ++i) {
            if (horiz.homogeneity[i] >= vert.homogeneity[i]) {
                out.r[i] = horiz.img.r[i];
                out.g[i] = horiz.img.g[i];
                out.b[i] = horiz.img.b[i];
            } else {
                out.r[i] = vert.img.r[i];
                out.g[i] = vert.img.g[i];
                out.b[i] = vert.img.b[i];
            }
        }
    }

    void refine_color_artifacts(Image& img, float maxVal) {
        std::vector<float> rg(img.width * img.height);
        std::vector<float> bg(img.width * img.height);

        for (int i = 0; i < img.width * img.height; ++i) {
            rg[i] = img.r[i] - img.g[i];
            bg[i] = img.b[i] - img.g[i];
        }

        cv::Mat rgMat(img.height, img.width, CV_32F, rg.data());
        cv::Mat bgMat(img.height, img.width, CV_32F, bg.data());
        cv::Mat rgMed, bgMed;
        cv::medianBlur(rgMat, rgMed, 3);
        cv::medianBlur(bgMat, bgMed, 3);

        for (int y = 0; y < img.height; ++y) {
            const float* rgRow = rgMed.ptr<float>(y);
            const float* bgRow = bgMed.ptr<float>(y);
            for (int x = 0; x < img.width; ++x) {
                int id = idx(x, y, img.width);
                img.r[id] = img.g[id] + rgRow[x];
                img.b[id] = img.g[id] + bgRow[x];
                img.r[id] = std::min(maxVal, std::max(0.0f, img.r[id]));
                img.b[id] = std::min(maxVal, std::max(0.0f, img.b[id]));
            }
        }
    }

    cv::Mat toMat(const Image& img, int raw_type) {
        cv::Mat f(img.height, img.width, CV_32FC3);
        for (int y = 0; y < img.height; ++y) {
            cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < img.width; ++x) {
                int id = idx(x, y, img.width);
                row[x][0] = img.b[id];
                row[x][1] = img.g[id];
                row[x][2] = img.r[id];
            }
        }

        cv::Mat dst;
        f.convertTo(dst, CV_MAKETYPE(CV_MAT_DEPTH(raw_type), 3));
        return dst;
    }
};

void demosiacAHD(const cv::Mat& raw, cv::Mat& dst, bayerPattern pattern) {
    AHDEngine engine;
    engine.process(raw, dst, pattern);
}