#pragma once

/**
 * sgbm_gpu.h - GPU Semi-Global Block Matching interface
 * 
 * SGBM (Semi-Global Block Matching) is a stereo matching algorithm that:
 * 1. Computes matching cost for each pixel at each disparity level
 * 2. Aggregates costs along multiple paths (semi-global optimization)
 * 3. Selects the disparity with minimum aggregated cost
 * 
 * This is the core of the stereo pipeline - we'll implement it in Phase 4.
 */

#include "stereo_params.h"
#include "sgbm_limits.h"
#include "cuda_buffer.h"
#include <cstdint>
#include <opencv2/core.hpp>

namespace stereo {

/**
 * GPU SGBM stereo matcher
 * 
 * Usage:
 *   StereoSGBM sgbm(params);
 *   sgbm.compute(leftRect, rightRect, disparity);
 */
class StereoSGBM {
public:
    explicit StereoSGBM(const StereoParams& params = StereoParams());
    ~StereoSGBM();
    
    /**
     * Compute disparity map from rectified stereo pair
     * 
     * @param left  Left rectified image (grayscale, CV_8UC1)
     * @param right Right rectified image (grayscale, CV_8UC1)
     * @param disparity Output disparity map (CV_16SC1, values in 1/16 pixels)
     */
    void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity);
    
    /**
     * Update algorithm parameters
     */
    void setParams(const StereoParams& params);
    const StereoParams& getParams() const { return params_; }
    
    /**
     * Get timing information from last compute() call
     */
    float getLastComputeTimeMs() const { return lastComputeTimeMs_; }
    float getCostComputeTimeMs() const { return costComputeTimeMs_; }
    float getAggregationTimeMs() const { return aggregationTimeMs_; }
    float getPostProcessTimeMs() const { return postProcessTimeMs_; }
    
private:
    StereoParams params_;
    
    // Image dimensions (stored for kernel launches)
    int width_ = 0;
    int height_ = 0;
    
    // GPU buffers
    CudaBuffer<unsigned char> d_left_, d_right_;
    CudaBuffer<unsigned char> d_leftFiltered_, d_rightFiltered_; // Sobel-prefiltered images for SAD
    // Census descriptors (one uint32 per pixel, 24 bits used) — allocated when matchingCostMode == Census
    CudaBuffer<uint32_t> d_leftCensus_, d_rightCensus_;
    CudaBuffer<unsigned char> d_costVolume_; // [H x W x D] matching costs (uint8, scaled up in aggregation)
    CudaBuffer<uint16_t> d_aggregatedCost_;  // [H x W x D] after SGM
    CudaBuffer<int16_t> d_disparity_;        // [H x W] output
    CudaBuffer<int16_t> d_disparityR_;       // [H x W] right-to-left for consistency
    CudaBuffer<int16_t> d_disparityTemp_;    // [H x W] temporary for post-processing
    
    // Timing
    float lastComputeTimeMs_ = 0;
    float costComputeTimeMs_ = 0;
    float aggregationTimeMs_ = 0;
    float postProcessTimeMs_ = 0;
    
    // CUDA events for timing
    cudaEvent_t startEvent_, stopEvent_;
    cudaStream_t stream_;
    
    // Internal methods
    void allocateBuffers(int width, int height);
    void computeCostVolume(const cv::Mat& left, const cv::Mat& right);
    void aggregateCosts();
    void selectDisparity();
    void postProcess();
};

} // namespace stereo
