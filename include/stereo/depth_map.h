#pragma once

/**
 * depth_map.h - Depth map generation and colorization
 * 
 * Converts disparity to depth using the formula:
 *   depth = (focal_length * baseline) / disparity
 */

#include "stereo_params.h"
#include "cuda_buffer.h"
#include <opencv2/core.hpp>

namespace stereo {

/**
 * Convert disparity map to depth map
 * 
 * @param disparity Input disparity (CV_16SC1, 1/16 pixel units)
 * @param depth Output depth in meters (CV_32FC1)
 * @param focalLength Focal length in pixels
 * @param baseline Distance between cameras in meters
 */
// disparityOffsetPx is added to the computed disparity (in pixels) before
// converting to depth. For Middlebury datasets, this corresponds to `doffs`.
void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                      double focalLength, double baseline,
                      double disparityOffsetPx = 0.0);

/**
 * GPU version of disparity to depth conversion
 */
void disparityToDepthGPU(const CudaBuffer<int16_t>& disparity,
                         CudaBuffer<float>& depth,
                         int width, int height,
                         float focalLength, float baseline,
                         float disparityOffsetPx = 0.0f);

/**
 * Colorize depth map using Turbo colormap
 * 
 * @param depth Input depth (CV_32FC1)
 * @param colorized Output BGR image (CV_8UC3)
 * @param minDepth Minimum depth for colormap (meters)
 * @param maxDepth Maximum depth for colormap (meters)
 */
void colorizeDepth(const cv::Mat& depth, cv::Mat& colorized,
                   float minDepth = 0.5f, float maxDepth = 10.0f);

/**
 * GPU colorization
 */
void colorizeDepthGPU(const CudaBuffer<float>& depth,
                      CudaBuffer<unsigned char>& colorized,
                      int width, int height,
                      float minDepth, float maxDepth);

} // namespace stereo
