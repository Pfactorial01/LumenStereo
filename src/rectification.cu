/**
 * rectification.cu - GPU-accelerated image rectification
 * 
 * Rectification aligns stereo images so that:
 * 1. Corresponding points lie on the same horizontal line (epipolar constraint)
 * 2. This reduces stereo matching from 2D search to 1D search
 * 
 * Implementation uses CUDA texture memory for:
 * - Hardware-accelerated bilinear interpolation
 * - Efficient 2D spatial caching
 * - Automatic boundary handling
 * 
 * The remapping process:
 * For each output pixel (x, y):
 *   1. Look up source coordinates (mapX[x,y], mapY[x,y])
 *   2. Sample source image at those coordinates with bilinear interpolation
 *   3. Write result to output pixel
 */

#include "stereo/rectification.h"
#include "stereo/common.h"

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>

namespace stereo {

// ============================================================================
// CUDA Texture Objects
// ============================================================================

/**
 * Create a CUDA texture object for an image
 * 
 * Texture objects are the modern way to use textures in CUDA (since CUDA 5.0).
 * They provide:
 * - Bindless textures (no limit on number of textures)
 * - Better performance than texture references
 * - More flexible configuration
 */
cudaTextureObject_t createTextureObject(
    const unsigned char* d_data,
    int width,
    int height,
    int channels
) {
    // Create channel descriptor
    cudaChannelFormatDesc channelDesc;
    if (channels == 1) {
        channelDesc = cudaCreateChannelDesc<unsigned char>();
    } else {
        // For multi-channel, we'll use uchar4
        channelDesc = cudaCreateChannelDesc<uchar4>();
    }
    
    // Create CUDA array
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
    
    // Copy data to CUDA array
    size_t pitch = width * channels * sizeof(unsigned char);
    CUDA_CHECK(cudaMemcpy2DToArray(
        cuArray, 0, 0,
        d_data, pitch,
        width * channels * sizeof(unsigned char), height,
        cudaMemcpyDeviceToDevice
    ));
    
    // Create texture resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    // Create texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;  // Clamp to edge
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;       // Bilinear interpolation
    texDesc.readMode = cudaReadModeNormalizedFloat;  // Return [0, 1] floats
    texDesc.normalizedCoords = false;                // Use pixel coordinates
    
    // Create texture object
    cudaTextureObject_t texObj;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    
    return texObj;
}

/**
 * Destroy texture object and free CUDA array
 */
void destroyTextureObject(cudaTextureObject_t texObj) {
    // Get the CUDA array from the texture object
    cudaResourceDesc resDesc;
    CUDA_CHECK(cudaGetTextureObjectResourceDesc(&resDesc, texObj));
    
    // Destroy texture object
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    
    // Free CUDA array
    if (resDesc.resType == cudaResourceTypeArray) {
        CUDA_CHECK(cudaFreeArray(resDesc.res.array.array));
    }
}

// ============================================================================
// Rectification Kernels
// ============================================================================

/**
 * GPU kernel for image remapping (grayscale)
 * 
 * Uses texture memory for bilinear interpolation.
 * The texture sampler automatically handles:
 * - Bilinear interpolation between pixels
 * - Boundary clamping
 * - Efficient caching
 */
__global__ void remapKernelGray(
    cudaTextureObject_t srcTex,
    const float* __restrict__ mapX,
    const float* __restrict__ mapY,
    unsigned char* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Get source coordinates from remap tables
    float srcX = mapX[idx];
    float srcY = mapY[idx];
    
    // Check bounds (texture will clamp, but we want black for out-of-bounds)
    if (srcX < 0 || srcX >= width || srcY < 0 || srcY >= height) {
        dst[idx] = 0;
        return;
    }
    
    // Sample texture with bilinear interpolation
    // tex2D automatically does bilinear interpolation!
    // We add 0.5 because tex2D samples at pixel centers
    float val = tex2D<float>(srcTex, srcX + 0.5f, srcY + 0.5f);
    
    // Convert from normalized [0,1] to [0,255]
    dst[idx] = static_cast<unsigned char>(val * 255.0f);
}

/**
 * GPU kernel for image remapping (BGR/RGB color)
 */
__global__ void remapKernelColor(
    cudaTextureObject_t srcTexR,
    cudaTextureObject_t srcTexG,
    cudaTextureObject_t srcTexB,
    const float* __restrict__ mapX,
    const float* __restrict__ mapY,
    unsigned char* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float srcX = mapX[idx];
    float srcY = mapY[idx];
    
    if (srcX < 0 || srcX >= width || srcY < 0 || srcY >= height) {
        dst[idx * 3 + 0] = 0;
        dst[idx * 3 + 1] = 0;
        dst[idx * 3 + 2] = 0;
        return;
    }
    
    // Sample each channel
    float r = tex2D<float>(srcTexR, srcX + 0.5f, srcY + 0.5f);
    float g = tex2D<float>(srcTexG, srcX + 0.5f, srcY + 0.5f);
    float b = tex2D<float>(srcTexB, srcX + 0.5f, srcY + 0.5f);
    
    dst[idx * 3 + 0] = static_cast<unsigned char>(b * 255.0f);
    dst[idx * 3 + 1] = static_cast<unsigned char>(g * 255.0f);
    dst[idx * 3 + 2] = static_cast<unsigned char>(r * 255.0f);
}

/**
 * Simple remapping kernel without texture (slower but simpler)
 * Uses manual bilinear interpolation
 */
__global__ void remapKernelSimple(
    const unsigned char* __restrict__ src,
    const float* __restrict__ mapX,
    const float* __restrict__ mapY,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float srcXf = mapX[idx];
    float srcYf = mapY[idx];
    
    // Bilinear interpolation
    int x0 = static_cast<int>(floorf(srcXf));
    int y0 = static_cast<int>(floorf(srcYf));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float dx = srcXf - x0;
    float dy = srcYf - y0;
    
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w10 = dx * (1.0f - dy);
    float w11 = dx * dy;
    
    // Row stride for multi-channel images
    int rowStride = width * channels;
    
    // Check which neighbors are in bounds
    bool valid00 = (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height);
    bool valid01 = (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height);
    bool valid10 = (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height);
    bool valid11 = (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height);
    
    for (int c = 0; c < channels; c++) {
        // Get pixel values (0 if out of bounds - matches OpenCV BORDER_CONSTANT)
        float v00 = valid00 ? static_cast<float>(src[y0 * rowStride + x0 * channels + c]) : 0.0f;
        float v01 = valid01 ? static_cast<float>(src[y1 * rowStride + x0 * channels + c]) : 0.0f;
        float v10 = valid10 ? static_cast<float>(src[y0 * rowStride + x1 * channels + c]) : 0.0f;
        float v11 = valid11 ? static_cast<float>(src[y1 * rowStride + x1 * channels + c]) : 0.0f;
        
        float val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        dst[y * rowStride + x * channels + c] = static_cast<unsigned char>(val + 0.5f);
    }
}

// ============================================================================
// StereoRectifier Implementation
// ============================================================================

StereoRectifier::StereoRectifier(const CameraParams& params) 
    : params_(params)
{
    // Check if images are pre-rectified (Middlebury dataset has identity transforms)
    if (params_.map0x.empty() || params_.map0y.empty()) {
        // No rectification maps provided - assume pre-rectified
        preRectified_ = true;
        return;
    }
    
    preRectified_ = false;
    
    // Upload rectification maps to GPU
    int mapSize = params_.width * params_.height;
    
    d_mapX0_.resize(mapSize);
    d_mapY0_.resize(mapSize);
    d_mapX1_.resize(mapSize);
    d_mapY1_.resize(mapSize);
    
    // OpenCV stores maps as CV_32FC1
    d_mapX0_.copyFrom(reinterpret_cast<const float*>(params_.map0x.data), mapSize);
    d_mapY0_.copyFrom(reinterpret_cast<const float*>(params_.map0y.data), mapSize);
    d_mapX1_.copyFrom(reinterpret_cast<const float*>(params_.map1x.data), mapSize);
    d_mapY1_.copyFrom(reinterpret_cast<const float*>(params_.map1y.data), mapSize);
}

void StereoRectifier::rectify(
    const cv::Mat& leftRaw, const cv::Mat& rightRaw,
    cv::Mat& leftRect, cv::Mat& rightRect
) {
    // If pre-rectified, just copy
    if (preRectified_) {
        leftRect = leftRaw.clone();
        rightRect = rightRaw.clone();
        return;
    }
    
    int width = leftRaw.cols;
    int height = leftRaw.rows;
    int channels = leftRaw.channels();
    
    // Allocate output
    leftRect.create(height, width, leftRaw.type());
    rightRect.create(height, width, rightRaw.type());
    
    // Upload input images to GPU
    CudaBuffer<unsigned char> d_leftSrc, d_rightSrc;
    CudaBuffer<unsigned char> d_leftDst, d_rightDst;
    
    size_t imageSize = static_cast<size_t>(width) * height * channels;
    
    d_leftSrc.resize(imageSize);
    d_rightSrc.resize(imageSize);
    d_leftDst.resize(imageSize);
    d_rightDst.resize(imageSize);
    
    d_leftSrc.copyFrom(leftRaw.data, imageSize);
    d_rightSrc.copyFrom(rightRaw.data, imageSize);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(divUp(width, 16), divUp(height, 16));
    
    // Use simple remap kernel (texture version requires more setup)
    remapKernelSimple<<<gridSize, blockSize>>>(
        d_leftSrc.data(),
        d_mapX0_.data(),
        d_mapY0_.data(),
        d_leftDst.data(),
        width, height, channels
    );
    CUDA_CHECK_KERNEL();
    
    remapKernelSimple<<<gridSize, blockSize>>>(
        d_rightSrc.data(),
        d_mapX1_.data(),
        d_mapY1_.data(),
        d_rightDst.data(),
        width, height, channels
    );
    CUDA_CHECK_KERNEL();
    
    // Download results
    d_leftDst.copyTo(leftRect.data, imageSize);
    d_rightDst.copyTo(rightRect.data, imageSize);
}

// ============================================================================
// Standalone GPU Rectification Functions
// ============================================================================

/**
 * Rectify a single image on GPU using precomputed maps
 */
void rectifyImageGPU(
    const cv::Mat& src,
    const cv::Mat& mapX,
    const cv::Mat& mapY,
    cv::Mat& dst
) {
    int width = src.cols;
    int height = src.rows;
    int channels = src.channels();
    
    // Ensure source is contiguous
    cv::Mat srcCont;
    if (!src.isContinuous()) {
        srcCont = src.clone();
    } else {
        srcCont = src;
    }
    
    dst.create(height, width, src.type());
    
    // Upload to GPU
    CudaBuffer<unsigned char> d_src, d_dst;
    CudaBuffer<float> d_mapX, d_mapY;
    
    size_t imageSize = static_cast<size_t>(width) * height * channels;
    size_t mapSize = static_cast<size_t>(width) * height;
    
    d_src.resize(imageSize);
    d_dst.resize(imageSize);
    d_mapX.resize(mapSize);
    d_mapY.resize(mapSize);
    
    d_src.copyFrom(srcCont.data, imageSize);
    d_mapX.copyFrom(reinterpret_cast<const float*>(mapX.data), mapSize);
    d_mapY.copyFrom(reinterpret_cast<const float*>(mapY.data), mapSize);
    
    // Initialize dst to zero
    d_dst.zero();
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(divUp(width, 16), divUp(height, 16));
    
    remapKernelSimple<<<gridSize, blockSize>>>(
        d_src.data(),
        d_mapX.data(),
        d_mapY.data(),
        d_dst.data(),
        width, height, channels
    );
    CUDA_CHECK_KERNEL();
    
    // Synchronize before download
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download
    d_dst.copyTo(dst.data, imageSize);
}

/**
 * Benchmark GPU vs CPU rectification
 */
void benchmarkRectification(
    const cv::Mat& src,
    const cv::Mat& mapX,
    const cv::Mat& mapY,
    int numIterations
) {
    cv::Mat dst;
    
    // Warm-up
    cv::remap(src, dst, mapX, mapY, cv::INTER_LINEAR);
    rectifyImageGPU(src, mapX, mapY, dst);
    
    // CPU benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; i++) {
        cv::remap(src, dst, mapX, mapY, cv::INTER_LINEAR);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    
    // GPU benchmark
    auto gpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; i++) {
        rectifyImageGPU(src, mapX, mapY, dst);
    }
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count() / numIterations;
    double gpuMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count() / numIterations;
    
    std::cout << "Rectification benchmark (" << src.cols << "x" << src.rows << "):\n";
    std::cout << "  CPU (OpenCV): " << cpuMs << " ms\n";
    std::cout << "  GPU (CUDA):   " << gpuMs << " ms\n";
    std::cout << "  Speedup:      " << (cpuMs / gpuMs) << "x\n";
}

} // namespace stereo
