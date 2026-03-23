/**
 * sgbm_gpu.cu - SGBM pipeline orchestration
 * 
 * Full stereo matching pipeline:
 * 1. Upload images to GPU
 * 2. Prefilter images (Sobel-x + clip, SAD mode only)
 * 3. Compute cost volume (SAD mean or Census Hamming, uint8)
 * 4. SGM cost aggregation (4/8 directions, costScale + adaptive P2)
 * 5. Disparity selection (WTA with uniqueness check)
 * 6. Post-processing (LR consistency, single median filter)
 * 7. Download result
 *
 * Key design: costs are stored as uint8 to save VRAM, then amplified by
 * costScale during aggregation so that P1/P2 penalties have the correct
 * ratio relative to the effective cost range.
 */

#include "stereo/sgbm_gpu.h"
#include "stereo/sgbm_limits.h"
#include "stereo/common.h"
#include <opencv2/imgproc.hpp>
#include <string>

#define TILE_WIDTH 32
#define TILE_HEIGHT 16

namespace stereo {
    __global__ void prefilterXSobel(
        const unsigned char* img, unsigned char* filtered,
        int width, int height, int cap);

    __global__ void computeCostSAD_naive(
        const unsigned char* left, const unsigned char* right,
        unsigned char* costVolume, int width, int height, int maxDisparity, int minDisparity, int blockRadius);
    
    __global__ void computeCostSAD_shared(
        const unsigned char* left, const unsigned char* right,
        unsigned char* costVolume, int width, int height, int maxDisparity, int minDisparity, int blockRadius);

    __global__ void buildCensus5x5(
        const unsigned char* img, uint32_t* census, int width, int height);

    __global__ void computeCostCensus_naive(
        const uint32_t* leftCensus, const uint32_t* rightCensus,
        unsigned char* costVolume, int width, int height, int maxDisparity, int minDisparity);
    
    __global__ void aggregateCostHorizontalLR(
        const unsigned char* costVolume, uint16_t* aggCost, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale);
    
    __global__ void aggregateCostHorizontalRL(
        const unsigned char* costVolume, uint16_t* aggCost, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale);
    
    __global__ void aggregateCostVerticalTB(
        const unsigned char* costVolume, uint16_t* aggCost, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale);
    
    __global__ void aggregateCostVerticalBT(
        const unsigned char* costVolume, uint16_t* aggCost, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale);
    
    __global__ void aggregateCostDiagonalTLBR(
        const unsigned char* costVolume, uint16_t* aggCost, uint16_t* L_diag, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale, int diagIdx);
    
    __global__ void aggregateCostDiagonalBRTL(
        const unsigned char* costVolume, uint16_t* aggCost, uint16_t* L_diag, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale, int diagIdx);
    
    __global__ void aggregateCostDiagonalTRBL(
        const unsigned char* costVolume, uint16_t* aggCost, uint16_t* L_diag, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale, int diagIdx);
    
    __global__ void aggregateCostDiagonalBLTR(
        const unsigned char* costVolume, uint16_t* aggCost, uint16_t* L_diag, const unsigned char* img,
        int width, int height, int maxDisparity, int P1, int P2, int costScale, int diagIdx);
    
    __global__ void selectDisparityWTA(
        const uint16_t* aggregatedCost, int16_t* disparity,
        int width, int height, int maxDisparity, int minDisparity, int uniquenessRatio);
    
    __global__ void selectDisparityRight(
        const uint16_t* aggregatedCost, int16_t* disparityR,
        int width, int height, int maxDisparity, int minDisparity);
    
    __global__ void consistencyCheck(
        int16_t* disparityL, const int16_t* disparityR,
        int width, int height, int maxDiff);
    
    __global__ void medianFilter3x3(
        const int16_t* input, int16_t* output, int width, int height);
    
    __global__ void speckleFilter(
        int16_t* disparity, int width, int height, int maxSpeckleSize, int maxDiff);
}

namespace stereo {

StereoSGBM::StereoSGBM(const StereoParams& params) 
    : params_(params)
    , stream_(nullptr)
{
    CUDA_CHECK(cudaEventCreate(&startEvent_));
    CUDA_CHECK(cudaEventCreate(&stopEvent_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

StereoSGBM::~StereoSGBM() {
    if (startEvent_) cudaEventDestroy(startEvent_);
    if (stopEvent_) cudaEventDestroy(stopEvent_);
    if (stream_) cudaStreamDestroy(stream_);
}

void StereoSGBM::setParams(const StereoParams& params) {
    params_ = params;
}

void StereoSGBM::allocateBuffers(int width, int height) {
    width_ = width;
    height_ = height;
    
    size_t numPixels = static_cast<size_t>(width) * height;
    int numDisparities = params_.maxDisparity - params_.minDisparity;
    size_t costVolumeSize = numPixels * numDisparities;
    
    d_left_.resize(numPixels);
    d_right_.resize(numPixels);
    d_leftFiltered_.resize(numPixels);
    d_rightFiltered_.resize(numPixels);
    if (params_.matchingCostMode == MatchingCostMode::Census) {
        d_leftCensus_.resize(numPixels);
        d_rightCensus_.resize(numPixels);
    } else {
        d_leftCensus_.resize(0);
        d_rightCensus_.resize(0);
    }
    d_costVolume_.resize(costVolumeSize);
    d_aggregatedCost_.resize(costVolumeSize);
    d_disparity_.resize(numPixels);
    d_disparityR_.resize(numPixels);
    d_disparityTemp_.resize(numPixels);
    
    d_aggregatedCost_.zero();
}

void StereoSGBM::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) {
    if (left.empty() || right.empty()) {
        throw StereoException("Input images are empty");
    }
    if (left.size() != right.size()) {
        throw StereoException("Left and right images must have same size");
    }
    
    int width = left.cols;
    int height = left.rows;

    const int numDisparities = params_.maxDisparity - params_.minDisparity;
    if (numDisparities < 1) {
        throw StereoException("Invalid disparity range: maxDisparity must be greater than minDisparity");
    }
    if (numDisparities > kSgbmMaxDisparityRange) {
        throw StereoException(
            "Disparity range (maxDisparity - minDisparity) exceeds GPU limit (" +
            std::to_string(kSgbmMaxDisparityRange) + "). Reduce the search range.");
    }
    
    cv::Mat leftGray, rightGray;
    if (left.channels() == 3) {
        cv::cvtColor(left, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right, rightGray, cv::COLOR_BGR2GRAY);
    } else {
        leftGray = left;
        rightGray = right;
    }
    
    CUDA_CHECK(cudaEventRecord(startEvent_, stream_));
    
    allocateBuffers(width, height);
    
    d_left_.copyFrom(leftGray.data, width * height);
    d_right_.copyFrom(rightGray.data, width * height);
    
    computeCostVolume(leftGray, rightGray);
    aggregateCosts();
    selectDisparity();
    postProcess();
    
    CUDA_CHECK(cudaEventRecord(stopEvent_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stopEvent_));
    CUDA_CHECK(cudaEventElapsedTime(&lastComputeTimeMs_, startEvent_, stopEvent_));
    
    disparity.create(height, width, CV_16SC1);
    d_disparity_.copyTo(reinterpret_cast<int16_t*>(disparity.data), width * height);
}

void StereoSGBM::computeCostVolume(const cv::Mat& left, const cv::Mat& right) {
    int numDisparities = params_.maxDisparity - params_.minDisparity;
    int blockRadius = params_.blockSize / 2;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream_));
    
    if (params_.matchingCostMode == MatchingCostMode::Census) {
        dim3 censusBlock(16, 16);
        dim3 censusGrid(divUp(width_, 16), divUp(height_, 16));
        buildCensus5x5<<<censusGrid, censusBlock, 0, stream_>>>(
            d_left_.data(), d_leftCensus_.data(), width_, height_);
        buildCensus5x5<<<censusGrid, censusBlock, 0, stream_>>>(
            d_right_.data(), d_rightCensus_.data(), width_, height_);
        CUDA_CHECK_KERNEL();
        computeCostCensus_naive<<<censusGrid, censusBlock, 0, stream_>>>(
            d_leftCensus_.data(),
            d_rightCensus_.data(),
            d_costVolume_.data(),
            width_,
            height_,
            numDisparities,
            params_.minDisparity);
        CUDA_CHECK_KERNEL();
    } else {
        // Prefilter images with Sobel-x + clip for illumination normalization
        dim3 pfBlock(16, 16);
        dim3 pfGrid(divUp(width_, 16), divUp(height_, 16));
        int cap = std::max(1, params_.preFilterCap);

        prefilterXSobel<<<pfGrid, pfBlock, 0, stream_>>>(
            d_left_.data(), d_leftFiltered_.data(), width_, height_, cap);
        prefilterXSobel<<<pfGrid, pfBlock, 0, stream_>>>(
            d_right_.data(), d_rightFiltered_.data(), width_, height_, cap);
        CUDA_CHECK_KERNEL();

        if (params_.useSharedMemory) {
            dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
            dim3 gridSize(divUp(width_, TILE_WIDTH), divUp(height_, TILE_HEIGHT));
            
            int padRadius = blockRadius;
            int leftTileW = TILE_WIDTH + 2 * padRadius;
            int leftTileH = TILE_HEIGHT + 2 * padRadius;
            int rightTileW = TILE_WIDTH + 3 * padRadius + params_.minDisparity + numDisparities;
            int rightTileH = TILE_HEIGHT + 2 * padRadius;
            size_t sharedMemSize = (leftTileW * leftTileH + rightTileW * rightTileH) * sizeof(unsigned char);
            
            if (sharedMemSize <= 48 * 1024) {
                computeCostSAD_shared<<<gridSize, blockSize, sharedMemSize, stream_>>>(
                    d_leftFiltered_.data(),
                    d_rightFiltered_.data(),
                    d_costVolume_.data(),
                    width_,
                    height_,
                    numDisparities,
                    params_.minDisparity,
                    blockRadius
                );
            } else {
                dim3 naiveBlock(16, 16);
                dim3 naiveGrid(divUp(width_, 16), divUp(height_, 16));
                computeCostSAD_naive<<<naiveGrid, naiveBlock, 0, stream_>>>(
                    d_leftFiltered_.data(),
                    d_rightFiltered_.data(),
                    d_costVolume_.data(),
                    width_,
                    height_,
                    numDisparities,
                    params_.minDisparity,
                    blockRadius
                );
            }
        } else {
            dim3 blockSize(16, 16);
            dim3 gridSize(divUp(width_, 16), divUp(height_, 16));
            
            computeCostSAD_naive<<<gridSize, blockSize, 0, stream_>>>(
                d_leftFiltered_.data(),
                d_rightFiltered_.data(),
                d_costVolume_.data(),
                width_,
                height_,
                numDisparities,
                params_.minDisparity,
                blockRadius
            );
        }
    }
    CUDA_CHECK_KERNEL();
    
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&costComputeTimeMs_, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void StereoSGBM::aggregateCosts() {
    int numDisparities = params_.maxDisparity - params_.minDisparity;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream_));
    
    d_aggregatedCost_.zero();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // Compute costScale: amplifies uint8 costs so the effective range is
    // comparable to P1/P2.  For SAD with prefilter (costs in [0, 2*cap]),
    // multiplying by blockSize² recovers the sum.  For Census (costs in
    // [0, 255]), we use a scale that gives a similar effective range.
    int costScale;
    if (params_.matchingCostMode == MatchingCostMode::SAD) {
        costScale = params_.blockSize * params_.blockSize;
    } else {
        costScale = std::max(1, (params_.blockSize * params_.blockSize * 2 * params_.preFilterCap) / 255);
    }
    
    const unsigned char* imgPtr = d_left_.data();
    
    int numRowBlocks = divUp(height_, 256);
    aggregateCostHorizontalLR<<<numRowBlocks, 256, 0, stream_>>>(
        d_costVolume_.data(),
        d_aggregatedCost_.data(),
        imgPtr,
        width_, height_, numDisparities,
        params_.P1, params_.P2, costScale
    );
    CUDA_CHECK_KERNEL();
    
    aggregateCostHorizontalRL<<<numRowBlocks, 256, 0, stream_>>>(
        d_costVolume_.data(),
        d_aggregatedCost_.data(),
        imgPtr,
        width_, height_, numDisparities,
        params_.P1, params_.P2, costScale
    );
    CUDA_CHECK_KERNEL();
    
    int numColBlocks = divUp(width_, 256);
    aggregateCostVerticalTB<<<numColBlocks, 256, 0, stream_>>>(
        d_costVolume_.data(),
        d_aggregatedCost_.data(),
        imgPtr,
        width_, height_, numDisparities,
        params_.P1, params_.P2, costScale
    );
    CUDA_CHECK_KERNEL();
    
    aggregateCostVerticalBT<<<numColBlocks, 256, 0, stream_>>>(
        d_costVolume_.data(),
        d_aggregatedCost_.data(),
        imgPtr,
        width_, height_, numDisparities,
        params_.P1, params_.P2, costScale
    );
    CUDA_CHECK_KERNEL();
    
    if (params_.numDirections >= 8) {
        CudaBuffer<uint16_t> d_diagTemp;
        d_diagTemp.resize(static_cast<size_t>(width_) * height_ * numDisparities);
        
        int maxDiag = width_ + height_ - 1;
        int maxDim = std::max(width_, height_);
        int diagBlocks = divUp(maxDim, 256);
        
        d_diagTemp.zero();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int diag = 0; diag < maxDiag; diag++) {
            aggregateCostDiagonalTLBR<<<diagBlocks, 256, 0, stream_>>>(
                d_costVolume_.data(),
                d_aggregatedCost_.data(),
                d_diagTemp.data(),
                imgPtr,
                width_, height_, numDisparities,
                params_.P1, params_.P2, costScale, diag
            );
        }
        CUDA_CHECK_KERNEL();
        
        d_diagTemp.zero();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int diag = 0; diag < maxDiag; diag++) {
            aggregateCostDiagonalBRTL<<<diagBlocks, 256, 0, stream_>>>(
                d_costVolume_.data(),
                d_aggregatedCost_.data(),
                d_diagTemp.data(),
                imgPtr,
                width_, height_, numDisparities,
                params_.P1, params_.P2, costScale, diag
            );
        }
        CUDA_CHECK_KERNEL();
        
        d_diagTemp.zero();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int diag = 0; diag < maxDiag; diag++) {
            aggregateCostDiagonalTRBL<<<diagBlocks, 256, 0, stream_>>>(
                d_costVolume_.data(),
                d_aggregatedCost_.data(),
                d_diagTemp.data(),
                imgPtr,
                width_, height_, numDisparities,
                params_.P1, params_.P2, costScale, diag
            );
        }
        CUDA_CHECK_KERNEL();
        
        d_diagTemp.zero();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int diag = 0; diag < maxDiag; diag++) {
            aggregateCostDiagonalBLTR<<<diagBlocks, 256, 0, stream_>>>(
                d_costVolume_.data(),
                d_aggregatedCost_.data(),
                d_diagTemp.data(),
                imgPtr,
                width_, height_, numDisparities,
                params_.P1, params_.P2, costScale, diag
            );
        }
        CUDA_CHECK_KERNEL();
    }
    
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&aggregationTimeMs_, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void StereoSGBM::selectDisparity() {
    int numDisparities = params_.maxDisparity - params_.minDisparity;
    
    dim3 blockSize(16, 16);
    dim3 gridSize(divUp(width_, 16), divUp(height_, 16));
    
    selectDisparityWTA<<<gridSize, blockSize, 0, stream_>>>(
        d_aggregatedCost_.data(),
        d_disparity_.data(),
        width_,
        height_,
        numDisparities,
        params_.minDisparity,
        params_.uniquenessRatio
    );
    CUDA_CHECK_KERNEL();
    
    selectDisparityRight<<<gridSize, blockSize, 0, stream_>>>(
        d_aggregatedCost_.data(),
        d_disparityR_.data(),
        width_,
        height_,
        numDisparities,
        params_.minDisparity
    );
    CUDA_CHECK_KERNEL();
}

void StereoSGBM::postProcess() {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream_));
    
    dim3 blockSize(16, 16);
    dim3 gridSize(divUp(width_, 16), divUp(height_, 16));
    
    if (params_.disp12MaxDiff >= 0) {
        consistencyCheck<<<gridSize, blockSize, 0, stream_>>>(
            d_disparity_.data(),
            d_disparityR_.data(),
            width_,
            height_,
            params_.disp12MaxDiff * 16
        );
        CUDA_CHECK_KERNEL();
    }
    
    if (params_.speckleWindowSize > 0) {
        int minSimilar = std::min(params_.speckleWindowSize, 25);
        speckleFilter<<<gridSize, blockSize, 0, stream_>>>(
            d_disparity_.data(),
            width_,
            height_,
            minSimilar,
            params_.speckleRange * 16
        );
        CUDA_CHECK_KERNEL();
    }
    
    // Single median pass (two passes over-smoothed the disparity)
    medianFilter3x3<<<gridSize, blockSize, 0, stream_>>>(
        d_disparity_.data(),
        d_disparityTemp_.data(),
        width_,
        height_
    );
    CUDA_CHECK_KERNEL();
    
    CUDA_CHECK(cudaMemcpyAsync(
        d_disparity_.data(),
        d_disparityTemp_.data(),
        width_ * height_ * sizeof(int16_t),
        cudaMemcpyDeviceToDevice,
        stream_
    ));
    
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&postProcessTimeMs_, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace stereo
