#include "blc.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

// // dynamic statistic
// blackLevels calcDynamicStatistic(const uint16_t* rawImage, imageInfo& info) {
//     // total sum and count
//     int64_t sumR=0, sumGr=0, sumGb=0, sumB=0;
//     int64_t cntR=0, cntGr=0, cntGb=0, cntB=0;
//
//     // check bo_rows validity
//     if (info.ob_rows > info.height) {
//         info.ob_rows = info.height;
//     }
//     if (info.ob_rows < 0) {
//         info.ob_rows = 0;
//     }
//     
//     // iterate the ob_rows
//     for (int y = 0; y < info.ob_rows; y++) {
//         bool ye = (y & 1) == 0;
//         int rowBase = y * info.width;
//         for (int x = 0; x < info.width; x++) {
//             bool xe = (x & 1) == 0;
//             uint16_t value = rawImage[rowBase + x];
//             if (ye && xe) {
//                 sumR += value;
//                 cntR++;
//             } else if (ye && !xe) {
//                 sumGr += value;
//                 cntGr++;
//             } else if (!ye && xe) {
//                 sumGb += value;
//                 cntGb++;
//             } else {
//                 sumB += value;
//                 cntB++;
//             }
//         }
//     }
//
//     // calculate the means of each channel
//     blackLevels bls{};
//     bls.r  = cntR  ? (float)(sumR  / (double)cntR)  : 0.0f;
//     bls.gr = cntGr ? (float)(sumGr / (double)cntGr) : 0.0f;
//     bls.gb = cntGb ? (float)(sumGb / (double)cntGb) : 0.0f;
//     bls.b  = cntB  ? (float)(sumB  / (double)cntB)  : 0.0f;
//
//     return bls;
// }

// correction
// black statistic = {135, 140, 145}
void applyBlc(uint16_t* rawImage, const imageInfo& info, const blackLevels& bls) {
    int height = info.height;
    int width = info.width;

    // convert to int saving performance
    // NOTE: guard against negative black levels (can happen with bad stats) to avoid uint16 wrap-around.
    auto clamp_black = [](float v) -> int {
        int iv = static_cast<int>(v + (v >= 0.0f ? 0.5f : -0.5f));
        return std::clamp(iv, 0, 65535);
    };
    int bl_r_int  = clamp_black(bls.r);
    int bl_gr_int = clamp_black(bls.gr);
    int bl_gb_int = clamp_black(bls.gb);
    int bl_b_int  = clamp_black(bls.b);

    // apply correction
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            int black = 0;

            bool ye = (y & 1) == 0;
            bool xe = (x & 1) == 0;

            // get raw[idx] and judge (r/ gr/ gb/ b)
            // choose the correct bl value
            if (ye && xe) {
                black = bl_r_int;
            } else if (ye && !xe) {
                black = bl_gr_int;
            } else if (!ye && xe) {
                black = bl_gb_int;
            } else {
                black = bl_b_int;
            }

            // value < 0 ? 0 : value
            int corrected = static_cast<int>(rawImage[idx]) - black;
            corrected = std::clamp(corrected, 0, 65535);
            rawImage[idx] = static_cast<uint16_t>(corrected);
        
        }
    }

    

}



