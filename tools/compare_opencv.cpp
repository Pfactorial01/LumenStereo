/**
 * compare_opencv.cpp - Compare our SGBM against OpenCV's implementation
 * 
 * This benchmark:
 * 1. Runs both implementations on the same images
 * 2. Compares quality (against ground truth if available)
 * 3. Compares performance (timing)
 * 
 * Usage:
 *   ./compare_opencv --middlebury <dataset_path>
 *   ./compare_opencv --synthetic <width> <height>
 */

#include "stereo/sgbm_gpu.h"
#include "stereo/sgbm_limits.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

/**
 * Load PFM file (Middlebury ground truth format)
 */
static bool hostIsLittleEndian() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}

static void byteswapFloat32Inplace(float& v) {
    uint32_t u;
    static_assert(sizeof(u) == sizeof(v), "float must be 32-bit");
    std::memcpy(&u, &v, sizeof(u));
    u = ((u & 0x000000FFu) << 24) |
        ((u & 0x0000FF00u) << 8) |
        ((u & 0x00FF0000u) >> 8) |
        ((u & 0xFF000000u) >> 24);
    std::memcpy(&v, &u, sizeof(u));
}

cv::Mat loadPFM(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return cv::Mat();

    std::string magic;
    file >> magic;
    if (magic != "Pf") return cv::Mat();

    int width = 0, height = 0;
    float scale = 0.0f;
    file >> width >> height >> scale;
    file.get();

    if (width <= 0 || height <= 0) return cv::Mat();

    const bool fileLittleEndian = (scale < 0.0f);
    const bool needSwap = (fileLittleEndian != hostIsLittleEndian());

    cv::Mat data(height, width, CV_32FC1);
    file.read(reinterpret_cast<char*>(data.data), static_cast<std::streamsize>(width) * height * sizeof(float));
    if (!file) return cv::Mat();

    if (needSwap) {
        for (int y = 0; y < data.rows; y++) {
            float* row = data.ptr<float>(y);
            for (int x = 0; x < data.cols; x++) {
                byteswapFloat32Inplace(row[x]);
            }
        }
    }

    cv::flip(data, data, 0);
    return data;
}

/**
 * Evaluate disparity against ground truth
 */
struct EvalResults {
    double mae;           // Mean Absolute Error
    double rms;           // Root Mean Square Error
    double badPixels1;    // % with error > 1px
    double badPixels2;    // % with error > 2px
    double badPixels4;    // % with error > 4px
    int validPixels;
};

EvalResults evaluate(const cv::Mat& computed, const cv::Mat& groundTruth, int maxDisparity) {
    EvalResults results = {0, 0, 0, 0, 0, 0};
    
    double sumAbsError = 0;
    double sumSqError = 0;
    int bad1 = 0, bad2 = 0, bad4 = 0;
    
    for (int y = 0; y < groundTruth.rows; y++) {
        for (int x = 0; x < groundTruth.cols; x++) {
            float gt = groundTruth.at<float>(y, x);
            
            // Skip invalid ground truth
            if (gt <= 0 || gt > 1000 || std::isinf(gt)) continue;
            
            // Get computed disparity (convert from 1/16 pixel)
            float comp;
            if (computed.type() == CV_16SC1) {
                int16_t raw = computed.at<int16_t>(y, x);
                if (raw <= 0) continue;
                comp = raw / 16.0f;
            } else {
                comp = computed.at<float>(y, x);
                if (comp <= 0) continue;
            }
            
            float error = std::abs(comp - gt);
            
            results.validPixels++;
            sumAbsError += error;
            sumSqError += error * error;
            
            if (error > 1.0f) bad1++;
            if (error > 2.0f) bad2++;
            if (error > 4.0f) bad4++;
        }
    }
    
    if (results.validPixels > 0) {
        results.mae = sumAbsError / results.validPixels;
        results.rms = std::sqrt(sumSqError / results.validPixels);
        results.badPixels1 = 100.0 * bad1 / results.validPixels;
        results.badPixels2 = 100.0 * bad2 / results.validPixels;
        results.badPixels4 = 100.0 * bad4 / results.validPixels;
    }
    
    return results;
}

void printResults(const std::string& name, const EvalResults& results, double timeMs) {
    std::cout << std::left << std::setw(15) << name << " | ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(8) << timeMs << " ms | ";
    std::cout << std::setw(6) << results.mae << " px | ";
    std::cout << std::setw(6) << results.badPixels2 << "% | ";
    std::cout << std::setw(8) << results.validPixels << "\n";
}

int main(int argc, char** argv) {
    std::cout << "=========================================================\n";
    std::cout << "   LumenStereo vs OpenCV SGBM Comparison\n";
    std::cout << "=========================================================\n\n";
    
    // Check for CUDA
    if (stereo::printDeviceInfo()) {
        std::cout << "\n";
    } else {
        std::cout << "Warning: CUDA not available, using CPU fallback\n\n";
    }
    
    // Parse arguments
    std::string datasetPath;
    int synthWidth = 640, synthHeight = 480;
    bool useSynthetic = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--middlebury" && i + 1 < argc) {
            datasetPath = argv[++i];
        } else if (arg == "--synthetic") {
            useSynthetic = true;
            if (i + 2 < argc) {
                synthWidth = std::stoi(argv[++i]);
                synthHeight = std::stoi(argv[++i]);
            }
        }
    }
    
    cv::Mat left, right, groundTruth;
    int minDisparity = 0;
    int maxDisparity = 128;
    int numDisparities = 128;  // maxDisparity - minDisparity
    
    if (!datasetPath.empty()) {
        // Load Middlebury dataset
        stereo::CameraParams camParams;
        stereo::loadMiddleburyCalib(datasetPath + "/calib.txt", camParams);
        
        left = cv::imread(datasetPath + "/im0.png", cv::IMREAD_GRAYSCALE);
        right = cv::imread(datasetPath + "/im1.png", cv::IMREAD_GRAYSCALE);
        groundTruth = loadPFM(datasetPath + "/disp0.pfm");
        
        // Middlebury `doffs` is an additive offset used in depth conversion (Z = fB/(disp+doffs)).
        // The ground-truth disparities are typically reported without `doffs`, so keep minDisparity=0.
        minDisparity = 0;
        numDisparities = stereo::middleburyMatcherMaxDisparity(camParams.numDisparities);
        maxDisparity = minDisparity + numDisparities;
        if (camParams.numDisparities > stereo::kSgbmMaxDisparityRange) {
            std::cerr << "Warning: calib ndisp=" << camParams.numDisparities
                      << " exceeds GPU limit; using " << stereo::kSgbmMaxDisparityRange << "\n";
        }
        
        if (left.empty() || right.empty()) {
            std::cerr << "Failed to load Middlebury images\n";
            return 1;
        }
    } else if (useSynthetic) {
        // Create synthetic images
        std::cout << "Using synthetic images: " << synthWidth << "x" << synthHeight << "\n\n";
        
        left.create(synthHeight, synthWidth, CV_8UC1);
        right.create(synthHeight, synthWidth, CV_8UC1);
        cv::randu(left, 0, 255);
        cv::randu(right, 0, 255);
        
        minDisparity = 0;
        numDisparities = 64;
        maxDisparity = 64;
    } else {
        std::cout << "Usage:\n";
        std::cout << "  ./compare_opencv --middlebury <dataset_path>\n";
        std::cout << "  ./compare_opencv --synthetic [width height]\n";
        return 1;
    }
    
    std::cout << "Image size: " << left.cols << "x" << left.rows << "\n";
    std::cout << "Disparity range: " << minDisparity << " - " << maxDisparity 
              << " (numDisparities=" << numDisparities << ")\n\n";
    
    // =========================================================================
    // Configure both implementations
    // =========================================================================
    
    int blockSize = 5;
    int P1 = 8 * blockSize * blockSize;
    int P2 = 32 * blockSize * blockSize;
    int numDirections = 4;  // Fair comparison: both use 4 directions
    
    // Our implementation
    stereo::StereoParams ourParams;
    ourParams.minDisparity = minDisparity;
    ourParams.maxDisparity = maxDisparity;
    ourParams.blockSize = blockSize;
    ourParams.P1 = P1;
    ourParams.P2 = P2;
    ourParams.numDirections = numDirections;
    ourParams.disp12MaxDiff = -1;  // Disable for comparison
    ourParams.uniquenessRatio = 0;
    ourParams.speckleWindowSize = 0;
    if (!datasetPath.empty()) {
        ourParams.matchingCostMode = stereo::MatchingCostMode::Census;
    }
    
    const int numRuns = 5;
    cv::Mat cvDisparitySaved;
    double cvAvgTime = 0;
    
    // OpenCV first, then release it — avoids peak VRAM (OpenCV + LumenStereo) on 6GB GPUs
    {
        cv::Ptr<cv::StereoSGBM> cvSGBM = cv::StereoSGBM::create(
            minDisparity,
            numDisparities,
            blockSize,
            P1,
            P2,
            -1,
            0,
            0,
            0,
            0,
            cv::StereoSGBM::MODE_HH4
        );
        cv::Mat cvDisparity;
        std::cout << "Warming up OpenCV SGBM...\n";
        cvSGBM->compute(left, right, cvDisparity);
        std::cout << "Running OpenCV " << numRuns << " iterations...\n";
        auto cvStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numRuns; i++) {
            cvSGBM->compute(left, right, cvDisparity);
        }
        auto cvEnd = std::chrono::high_resolution_clock::now();
        cvAvgTime = std::chrono::duration<double, std::milli>(cvEnd - cvStart).count() / numRuns;
        cvDisparity.copyTo(cvDisparitySaved);
    }
    cudaDeviceSynchronize();
    
    stereo::StereoSGBM ourSGBM(ourParams);
    cv::Mat ourDisparity;
    std::cout << "Warming up LumenStereo...\n";
    ourSGBM.compute(left, right, ourDisparity);
    std::cout << "Running LumenStereo " << numRuns << " iterations...\n\n";
    double ourTotalTime = 0;
    for (int i = 0; i < numRuns; i++) {
        ourSGBM.compute(left, right, ourDisparity);
        ourTotalTime += ourSGBM.getLastComputeTimeMs();
    }
    double ourAvgTime = ourTotalTime / numRuns;
    
    // =========================================================================
    // Quality evaluation (if ground truth available)
    // =========================================================================
    
    EvalResults ourResults = {0}, cvResults = {0};
    
    if (!groundTruth.empty()) {
        ourResults = evaluate(ourDisparity, groundTruth, maxDisparity);
        cvResults = evaluate(cvDisparitySaved, groundTruth, maxDisparity);
    }
    
    // =========================================================================
    // Print results
    // =========================================================================
    
    std::cout << "=========================================================\n";
    std::cout << "                      RESULTS\n";
    std::cout << "=========================================================\n";
    std::cout << std::left << std::setw(15) << "Method" << " | ";
    std::cout << std::setw(11) << "Time" << " | ";
    std::cout << std::setw(9) << "MAE" << " | ";
    std::cout << std::setw(9) << "Bad>2px" << " | ";
    std::cout << "Valid\n";
    std::cout << "---------------------------------------------------------\n";
    
    printResults("LumenStereo", ourResults, ourAvgTime);
    printResults("OpenCV SGBM", cvResults, cvAvgTime);
    
    std::cout << "---------------------------------------------------------\n";
    
    // Speedup
    double speedup = cvAvgTime / ourAvgTime;
    std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) << speedup << "x ";
    if (speedup > 1) {
        std::cout << "(LumenStereo is faster)\n";
    } else {
        std::cout << "(OpenCV is faster)\n";
    }
    
    // Timing breakdown for our implementation
    std::cout << "\nLumenStereo timing breakdown:\n";
    std::cout << "  Cost computation:   " << ourSGBM.getCostComputeTimeMs() << " ms\n";
    std::cout << "  SGM aggregation:    " << ourSGBM.getAggregationTimeMs() << " ms\n";
    std::cout << "  Post-processing:    " << ourSGBM.getPostProcessTimeMs() << " ms\n";
    
    // FPS comparison
    std::cout << "\nFrames per second:\n";
    std::cout << "  LumenStereo: " << std::fixed << std::setprecision(1) << (1000.0 / ourAvgTime) << " FPS\n";
    std::cout << "  OpenCV SGBM: " << (1000.0 / cvAvgTime) << " FPS\n";
    
    // =========================================================================
    // Save outputs
    // =========================================================================
    
    std::cout << "\nSaving outputs...\n";
    
    // Convert CV_16SC1 (1/16 px) to preview: use the SAME linear scale for both so
    // images are comparable. (Bug we fixed: OpenCV was wrongly scaled by LumenStereo's
    // min-max, which crushed or blew out contrast on one or both maps.)
    cv::Mat ourVis, cvVis;
    cv::Mat ourFloat, cvFloat;
    ourDisparity.convertTo(ourFloat, CV_32F, 1.0 / 16.0);
    cvDisparitySaved.convertTo(cvFloat, CV_32F, 1.0 / 16.0);
    // Map [0, dispVizMax] -> [0, 255]; invalid (<=0) stay dark after saturate
    const float dispVizMax = static_cast<float>(std::max(1, maxDisparity));
    const double scale8u = 255.0 / static_cast<double>(dispVizMax);
    ourFloat.convertTo(ourVis, CV_8U, scale8u);
    cvFloat.convertTo(cvVis, CV_8U, scale8u);
    ourVis.setTo(0, ourFloat <= 0);
    cvVis.setTo(0, cvFloat <= 0);
    
    cv::applyColorMap(ourVis, ourVis, cv::COLORMAP_TURBO);
    cv::applyColorMap(cvVis, cvVis, cv::COLORMAP_TURBO);
    
    cv::imwrite("lumenstereo_disparity.png", ourVis);
    cv::imwrite("opencv_disparity.png", cvVis);
    
    std::cout << "  lumenstereo_disparity.png\n";
    std::cout << "  opencv_disparity.png\n";
    
    std::cout << "\nDone!\n";
    
    return 0;
}
