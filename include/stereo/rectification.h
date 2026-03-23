#pragma once

/**
 * rectification.h - GPU-accelerated stereo rectification
 * 
 * Rectification transforms stereo images so that:
 * 1. Epipolar lines become horizontal (corresponding points on same row)
 * 2. Images are undistorted (lens distortion removed)
 * 3. Principal points are aligned
 * 
 * This enables 1D stereo matching instead of 2D search.
 * 
 * Implementation uses CUDA for GPU acceleration:
 * - Bilinear interpolation via texture memory or manual computation
 * - Each pixel is independent → highly parallel
 * - Typical speedup: 5-10x vs CPU
 */

#include "stereo_params.h"
#include "cuda_buffer.h"
#include <opencv2/core.hpp>
#include <chrono>

namespace stereo {

/**
 * GPU-accelerated stereo rectifier
 * 
 * Usage:
 *   // Create rectifier with calibration parameters
 *   StereoRectifier rectifier(cameraParams);
 *   
 *   // Check if already rectified (e.g., Middlebury dataset)
 *   if (!rectifier.isPreRectified()) {
 *       rectifier.rectify(leftRaw, rightRaw, leftRect, rightRect);
 *   }
 * 
 * The rectifier precomputes and uploads remap tables to GPU for efficiency.
 * Multiple rectify() calls reuse the same GPU-resident maps.
 */
class StereoRectifier {
public:
    /**
     * Create rectifier from calibration parameters
     * 
     * If params.map0x/map0y are empty, assumes images are pre-rectified.
     * Otherwise, uploads rectification maps to GPU memory.
     */
    explicit StereoRectifier(const CameraParams& params);
    
    /**
     * Rectify a stereo image pair
     * 
     * @param leftRaw   Raw left camera image
     * @param rightRaw  Raw right camera image
     * @param leftRect  Output rectified left image
     * @param rightRect Output rectified right image
     * 
     * If isPreRectified() is true, just copies input to output.
     */
    void rectify(const cv::Mat& leftRaw, const cv::Mat& rightRaw,
                 cv::Mat& leftRect, cv::Mat& rightRect);
    
    /**
     * Check if images are already rectified
     * 
     * Middlebury datasets are pre-rectified, so no processing needed.
     */
    bool isPreRectified() const { return preRectified_; }
    
    /**
     * Get the camera parameters
     */
    const CameraParams& getParams() const { return params_; }
    
private:
    CameraParams params_;
    bool preRectified_ = false;
    
    // GPU buffers for rectification maps
    // These map output (x,y) → source (srcX, srcY)
    CudaBuffer<float> d_mapX0_, d_mapY0_;  // Left camera maps
    CudaBuffer<float> d_mapX1_, d_mapY1_;  // Right camera maps
};

/**
 * Rectify a single image using GPU
 * 
 * Standalone function for one-off rectification.
 * For repeated rectification, use StereoRectifier class instead.
 */
void rectifyImageGPU(
    const cv::Mat& src,
    const cv::Mat& mapX,
    const cv::Mat& mapY,
    cv::Mat& dst
);

/**
 * Benchmark GPU vs CPU rectification
 * 
 * Runs both implementations and prints timing comparison.
 */
void benchmarkRectification(
    const cv::Mat& src,
    const cv::Mat& mapX,
    const cv::Mat& mapY,
    int numIterations = 100
);

} // namespace stereo
