/**
 * evaluate.cpp - Evaluate disparity against Middlebury ground truth
 * 
 * Metrics:
 * - Bad pixel %: Percentage of pixels with |error| > threshold
 * - RMS: Root mean square error
 * - MAE: Mean absolute error
 */

#include "stereo/sgbm_gpu.h"
#include "stereo/sgbm_limits.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

/**
 * Load PFM file (Middlebury ground truth format)
 * PFM stores 32-bit floats, row-major, bottom-to-top
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
    if (!file.is_open()) {
        std::cerr << "Failed to open PFM: " << path << "\n";
        return cv::Mat();
    }

    std::string magic;
    file >> magic;
    if (magic != "Pf") {
        std::cerr << "Invalid PFM magic: " << magic << " (expected Pf)\n";
        return cv::Mat();
    }

    int width = 0, height = 0;
    float scale = 0.0f;
    file >> width >> height >> scale;
    file.get(); // consume single whitespace/newline before binary data

    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid PFM dimensions\n";
        return cv::Mat();
    }

    // PFM endianness: negative scale means little-endian.
    bool fileLittleEndian = (scale < 0.0f);
    const bool needSwap = (fileLittleEndian != hostIsLittleEndian());

    cv::Mat data(height, width, CV_32FC1);
    file.read(reinterpret_cast<char*>(data.data), static_cast<std::streamsize>(width) * height * sizeof(float));
    if (!file) {
        std::cerr << "Failed to read PFM payload\n";
        return cv::Mat();
    }

    if (needSwap) {
        for (int y = 0; y < data.rows; y++) {
            float* row = data.ptr<float>(y);
            for (int x = 0; x < data.cols; x++) {
                byteswapFloat32Inplace(row[x]);
            }
        }
    }

    // PFM is stored bottom-to-top, flip it
    cv::flip(data, data, 0);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--middlebury") {
        std::cout << "Usage: ./evaluate --middlebury <dataset_path>\n";
        return 1;
    }

    std::string datasetPath = argv[2];
    
    try {
        // Load calibration
        stereo::CameraParams camParams;
        stereo::loadMiddleburyCalib(datasetPath + "/calib.txt", camParams);
        
        // Load images
        std::cout << "Loading images...\n";
        cv::Mat left = cv::imread(datasetPath + "/im0.png", cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(datasetPath + "/im1.png", cv::IMREAD_GRAYSCALE);
        
        if (left.empty() || right.empty()) {
            std::cerr << "Failed to load images\n";
            return 1;
        }
        
        // Load ground truth
        std::cout << "Loading ground truth...\n";
        cv::Mat gtDisp = loadPFM(datasetPath + "/disp0.pfm");
        if (gtDisp.empty()) {
            std::cerr << "Failed to load ground truth\n";
            return 1;
        }
        
        std::cout << "Ground truth size: " << gtDisp.cols << "x" << gtDisp.rows << "\n";
        
        // Analyze ground truth
        double gtMin, gtMax;
        cv::minMaxLoc(gtDisp, &gtMin, &gtMax);
        std::cout << "Ground truth disparity range: " << gtMin << " - " << gtMax << "\n";
        
        // Compute disparity with our SGBM (use Middlebury disparity range if available)
        std::cout << "\nComputing disparity with our SGBM...\n";
        
        stereo::StereoParams params;
        // Middlebury `doffs` is used for depth conversion, not as a matcher offset.
        params.minDisparity = 0;
        params.maxDisparity = stereo::middleburyMatcherMaxDisparity(camParams.numDisparities);
        if (camParams.numDisparities > stereo::kSgbmMaxDisparityRange) {
            std::cerr << "Warning: calib ndisp=" << camParams.numDisparities
                      << " exceeds GPU limit; using " << stereo::kSgbmMaxDisparityRange << "\n";
        }
        std::cout << "SGBM disparity search: min=" << params.minDisparity
                  << " max=" << params.maxDisparity
                  << " (ndisp=" << (params.maxDisparity - params.minDisparity) << ")\n";
        params.blockSize = 5;
        params.numDirections = 4;
        params.P1 = 200;
        params.P2 = 1600;                // Quality: smooth surfaces, fewer stripes
        params.uniquenessRatio = 5;
        params.disp12MaxDiff = 3;
        params.speckleWindowSize = 80;
        params.speckleRange = 24;
        params.useSharedMemory = true;
        // Census + Hamming cost (Step 4): stronger on Middlebury than raw SAD
        params.matchingCostMode = stereo::MatchingCostMode::Census;
        
        stereo::StereoSGBM sgbm(params);
        cv::Mat disparity;
        sgbm.compute(left, right, disparity);
        
        std::cout << "Compute time: " << sgbm.getLastComputeTimeMs() << " ms\n";
        
        // Convert our disparity from 1/16 pixel units to float pixels
        cv::Mat ourDisp;
        disparity.convertTo(ourDisp, CV_32F, 1.0 / 16.0);
        
        // Evaluate
        std::cout << "\n=== EVALUATION RESULTS ===\n";
        
        int validPixels = 0;
        int badPixels_1 = 0;   // |error| > 1 pixel
        int badPixels_2 = 0;   // |error| > 2 pixels
        int badPixels_4 = 0;   // |error| > 4 pixels
        double sumSqError = 0;
        double sumAbsError = 0;
        std::vector<float> errors;
        
        for (int y = 0; y < gtDisp.rows; y++) {
            for (int x = 0; x < gtDisp.cols; x++) {
                float gt = gtDisp.at<float>(y, x);
                float ours = ourDisp.at<float>(y, x);
                
                // Skip invalid ground truth (inf values) and outside our range
                if (gt <= 0 || gt > 1000 || std::isinf(gt)) continue;
                
                // Skip if our disparity is invalid
                if (ours <= 0) continue;
                
                // Compute error
                float error = std::abs(ours - gt);
                
                validPixels++;
                errors.push_back(error);
                sumAbsError += error;
                sumSqError += error * error;
                
                if (error > 1.0f) badPixels_1++;
                if (error > 2.0f) badPixels_2++;
                if (error > 4.0f) badPixels_4++;
            }
        }
        
        if (validPixels == 0) {
            std::cout << "No valid pixels to evaluate!\n";
            return 1;
        }
        
        // Calculate metrics
        double mae = sumAbsError / validPixels;
        double rms = std::sqrt(sumSqError / validPixels);
        double bad1 = 100.0 * badPixels_1 / validPixels;
        double bad2 = 100.0 * badPixels_2 / validPixels;
        double bad4 = 100.0 * badPixels_4 / validPixels;
        
        // Calculate percentiles
        std::sort(errors.begin(), errors.end());
        float err50 = errors[errors.size() * 50 / 100];
        float err90 = errors[errors.size() * 90 / 100];
        float err95 = errors[errors.size() * 95 / 100];
        float err99 = errors[errors.size() * 99 / 100];
        
        std::cout << "Valid pixels evaluated: " << validPixels 
                  << " (" << (100.0 * validPixels / (gtDisp.rows * gtDisp.cols)) << "%)\n\n";
        
        std::cout << "Error Metrics:\n";
        std::cout << "  MAE (Mean Absolute Error): " << mae << " pixels\n";
        std::cout << "  RMS (Root Mean Square):    " << rms << " pixels\n";
        std::cout << "\n";
        
        std::cout << "Bad Pixel Percentage:\n";
        std::cout << "  Bad > 1px: " << bad1 << "%\n";
        std::cout << "  Bad > 2px: " << bad2 << "%\n";
        std::cout << "  Bad > 4px: " << bad4 << "%\n";
        std::cout << "\n";
        
        std::cout << "Error Percentiles:\n";
        std::cout << "  50th percentile: " << err50 << " px\n";
        std::cout << "  90th percentile: " << err90 << " px\n";
        std::cout << "  95th percentile: " << err95 << " px\n";
        std::cout << "  99th percentile: " << err99 << " px\n";
        std::cout << "\n";
        
        // Quality assessment
        std::cout << "=== QUALITY ASSESSMENT ===\n";
        if (bad2 < 5) {
            std::cout << "EXCELLENT: < 5% bad pixels at 2px threshold\n";
        } else if (bad2 < 10) {
            std::cout << "GOOD: 5-10% bad pixels at 2px threshold\n";
        } else if (bad2 < 20) {
            std::cout << "MODERATE: 10-20% bad pixels at 2px threshold\n";
        } else if (bad2 < 40) {
            std::cout << "POOR: 20-40% bad pixels at 2px threshold\n";
        } else {
            std::cout << "VERY POOR: > 40% bad pixels at 2px threshold\n";
        }
        
        std::cout << "\nNote: Matching cost is Census (5x5) + SGM (see README roadmap).\n";
        
        // Compare to typical results
        std::cout << "\nTypical Middlebury results for reference:\n";
        std::cout << "  OpenCV SGBM:    ~5-10% bad pixels at 2px\n";
        std::cout << "  State-of-art:   ~2-3% bad pixels at 2px\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
