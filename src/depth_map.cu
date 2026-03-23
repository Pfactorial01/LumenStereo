/**
 * depth_map.cu - Depth map generation and colorization
 */

#include "stereo/depth_map.h"
#include "stereo/common.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace stereo {

// ============================================================================
// TURBO COLORMAP
// ============================================================================
// Pre-computed Turbo colormap values (256 entries)
// Perceptually uniform, better than jet/rainbow for depth visualization

__constant__ unsigned char turboR[256] = {
    48,50,51,52,54,55,56,57,59,60,61,62,63,64,66,67,68,69,70,71,72,73,74,75,76,
    77,78,79,80,81,82,83,84,85,86,87,88,88,89,90,91,92,93,93,94,95,96,96,97,98,
    99,99,100,101,101,102,103,103,104,105,105,106,106,107,108,108,109,109,110,
    110,111,111,112,112,113,113,114,114,115,115,116,116,116,117,117,118,118,118,
    119,119,120,120,120,121,121,121,122,122,122,123,123,123,124,124,124,125,125,
    125,125,126,126,126,126,127,127,127,127,128,128,128,128,129,129,129,129,129,
    130,130,130,130,130,131,131,131,131,131,131,132,132,132,132,132,132,133,133,
    133,133,133,133,133,134,134,134,134,134,134,134,134,135,135,135,135,135,135,
    135,135,135,136,136,136,136,136,136,136,136,136,136,137,137,137,137,137,137,
    137,137,137,137,137,137,138,138,138,138,138,138,138,138,138,138,138,138,138,
    138,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,140,
    140,140,140,140,140,140,140,140,140,140,140,140,140,140,140,140,140,140,140,
    141,141,141,141,141,141,141
};

__constant__ unsigned char turboG[256] = {
    18,21,24,27,30,33,35,38,41,43,46,48,51,53,55,58,60,62,64,66,68,70,72,74,76,
    78,80,81,83,85,87,88,90,91,93,94,96,97,99,100,102,103,104,106,107,108,109,
    111,112,113,114,115,117,118,119,120,121,122,123,124,125,126,127,128,129,130,
    131,132,133,134,135,136,137,137,138,139,140,141,141,142,143,144,145,145,146,
    147,148,148,149,150,150,151,152,152,153,154,154,155,156,156,157,157,158,159,
    159,160,160,161,161,162,163,163,164,164,165,165,166,166,167,167,168,168,169,
    169,170,170,171,171,172,172,172,173,173,174,174,175,175,175,176,176,177,177,
    177,178,178,179,179,179,180,180,180,181,181,182,182,182,183,183,183,184,184,
    184,185,185,185,186,186,186,187,187,187,188,188,188,189,189,189,189,190,190,
    190,191,191,191,192,192,192,192,193,193,193,194,194,194,194,195,195,195,195,
    196,196,196,196,197,197,197,197,198,198,198,198,199,199,199,199,200,200,200,
    200,200,201,201,201,201,202,202,202,202,202,203,203,203,203,203,204,204,204,
    204,204,205,205,205,205,205
};

__constant__ unsigned char turboB[256] = {
    107,111,115,119,123,127,131,135,138,142,145,149,152,155,158,161,164,167,170,
    172,175,177,180,182,184,186,188,190,192,194,196,197,199,200,202,203,205,206,
    207,208,209,210,211,212,213,214,215,215,216,217,217,218,218,219,219,220,220,
    220,221,221,221,222,222,222,222,223,223,223,223,223,223,223,224,224,224,224,
    224,224,224,224,224,224,224,224,224,223,223,223,223,223,223,223,222,222,222,
    222,221,221,221,221,220,220,220,219,219,219,218,218,217,217,216,216,215,215,
    214,214,213,213,212,212,211,210,210,209,209,208,207,207,206,205,204,204,203,
    202,201,201,200,199,198,197,196,196,195,194,193,192,191,190,189,188,187,186,
    185,184,183,182,181,180,179,178,177,176,175,174,173,171,170,169,168,167,165,
    164,163,162,160,159,158,157,155,154,153,151,150,149,147,146,145,143,142,140,
    139,138,136,135,133,132,130,129,127,126,124,123,121,120,118,117,115,114,112,
    111,109,108,106,105,103,102,100,99,97,96,94,93,91,90,88,87,85,84,82,81,79,
    78,76,75,73,72,70,69,67
};

// ============================================================================
// CUDA KERNELS
// ============================================================================

/**
 * Convert disparity to depth
 * depth = (f * B) / d
 */
__global__ void disparityToDepthKernel(
    const int16_t* __restrict__ disparity,
    float* __restrict__ depth,
    int width,
    int height,
    float focalLength,
    float baseline,
    float disparityOffsetPx
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int16_t d = disparity[idx];
    
    if (d <= 0) {
        // Invalid disparity
        depth[idx] = 0.0f;
    } else {
        // Convert from 1/16 pixel units to pixels
        float dispFloat = static_cast<float>(d) / 16.0f;
        // Middlebury-style offset (doffs) support:
        // depth = (f * B) / (disp + offset)
        float denom = dispFloat + disparityOffsetPx;
        depth[idx] = (denom > 0.0f) ? ((focalLength * baseline) / denom) : 0.0f;
    }
}

/**
 * Colorize depth using Turbo colormap
 */
__global__ void colorizeDepthKernel(
    const float* __restrict__ depth,
    unsigned char* __restrict__ colorized,  // RGB interleaved
    int width,
    int height,
    float minDepth,
    float maxDepth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float d = depth[idx];
    
    // Map depth to [0, 255]
    int colorIdx;
    if (d <= 0.0f || d > maxDepth * 2) {
        // Invalid depth - black
        colorIdx = 0;
    } else {
        // Clamp and normalize
        float normalized = (d - minDepth) / (maxDepth - minDepth);
        normalized = fmaxf(0.0f, fminf(1.0f, normalized));
        // Invert: close = warm (red), far = cool (blue)
        normalized = 1.0f - normalized;
        colorIdx = static_cast<int>(normalized * 255.0f);
    }
    
    // Write RGB
    int outIdx = idx * 3;
    colorized[outIdx + 0] = turboR[colorIdx];
    colorized[outIdx + 1] = turboG[colorIdx];
    colorized[outIdx + 2] = turboB[colorIdx];
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                      double focalLength, double baseline,
                      double disparityOffsetPx) {
    depth.create(disparity.size(), CV_32FC1);
    
    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            int16_t d = disparity.at<int16_t>(y, x);
            if (d <= 0) {
                depth.at<float>(y, x) = 0.0f;
            } else {
                float dispFloat = static_cast<float>(d) / 16.0f;
                float denom = dispFloat + static_cast<float>(disparityOffsetPx);
                depth.at<float>(y, x) = (denom > 0.0f)
                    ? static_cast<float>(focalLength * baseline / denom)
                    : 0.0f;
            }
        }
    }
}

void disparityToDepthGPU(const CudaBuffer<int16_t>& disparity,
                          CudaBuffer<float>& depth,
                          int width, int height,
                          float focalLength, float baseline,
                          float disparityOffsetPx) {
    depth.resize(width * height);
    
    dim3 blockSize(16, 16);
    dim3 gridSize(divUp(width, 16), divUp(height, 16));
    
    disparityToDepthKernel<<<gridSize, blockSize>>>(
        disparity.data(),
        depth.data(),
        width, height,
        focalLength, baseline,
        disparityOffsetPx
    );
    CUDA_CHECK_KERNEL();
}

void colorizeDepth(const cv::Mat& depth, cv::Mat& colorized,
                   float minDepth, float maxDepth) {
    colorized.create(depth.size(), CV_8UC3);
    
    // Simple CPU implementation using OpenCV
    cv::Mat normalized;
    depth.convertTo(normalized, CV_32F);
    normalized = (normalized - minDepth) / (maxDepth - minDepth);
    normalized = 1.0f - normalized;  // Invert
    cv::threshold(normalized, normalized, 0, 0, cv::THRESH_TOZERO);
    cv::threshold(normalized, normalized, 1, 1, cv::THRESH_TRUNC);
    normalized.convertTo(normalized, CV_8U, 255);
    
    cv::applyColorMap(normalized, colorized, cv::COLORMAP_TURBO);
}

void colorizeDepthGPU(const CudaBuffer<float>& depth,
                      CudaBuffer<unsigned char>& colorized,
                      int width, int height,
                      float minDepth, float maxDepth) {
    colorized.resize(width * height * 3);
    
    dim3 blockSize(16, 16);
    dim3 gridSize(divUp(width, 16), divUp(height, 16));
    
    colorizeDepthKernel<<<gridSize, blockSize>>>(
        depth.data(),
        colorized.data(),
        width, height,
        minDepth, maxDepth
    );
    CUDA_CHECK_KERNEL();
}

} // namespace stereo
