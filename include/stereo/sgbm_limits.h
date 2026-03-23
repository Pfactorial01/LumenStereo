#pragma once

/**
 * GPU SGBM compile-time limits (must match device stack buffers in sgbm_kernels.cu).
 * numDisparities = maxDisparity - minDisparity must not exceed this value.
 */
namespace stereo {

inline constexpr int kSgbmMaxDisparityRange = 512;

/**
 * Middlebury calib provides ndisp = number of disparity levels to search (with minDisparity = 0).
 * Use full ndisp for fair GT comparison; clamp to kSgbmMaxDisparityRange for GPU limits.
 *
 * @param calibNumDisparities  ndisp from calib.txt (0 if unknown)
 * @param fallbackWhenMissing  used when calibNumDisparities <= 0
 */
inline int middleburyMatcherMaxDisparity(int calibNumDisparities,
                                         int fallbackWhenMissing = 96) {
    const int ndisp = (calibNumDisparities > 0) ? calibNumDisparities : fallbackWhenMissing;
    return (ndisp > kSgbmMaxDisparityRange) ? kSgbmMaxDisparityRange : ndisp;
}

} // namespace stereo
