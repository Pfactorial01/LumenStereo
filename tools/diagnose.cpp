/**
 * diagnose.cpp - Diagnostic tool to analyze disparity/depth results
 */

#include "stereo/sgbm_gpu.h"
#include "stereo/sgbm_limits.h"
#include "stereo/depth_map.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--middlebury") {
        std::cout << "Usage: ./diagnose --middlebury <dataset_path>\n";
        return 1;
    }
    
    try {
        std::string datasetPath = argv[2];
        
        // Load calibration
        stereo::CameraParams camParams;
        stereo::loadMiddleburyCalib(datasetPath + "/calib.txt", camParams);
        
        // Load images
        cv::Mat leftImg = cv::imread(datasetPath + "/im0.png", cv::IMREAD_GRAYSCALE);
        cv::Mat rightImg = cv::imread(datasetPath + "/im1.png", cv::IMREAD_GRAYSCALE);
        
        if (leftImg.empty()) {
            std::cerr << "Failed to load images\n";
            return 1;
        }
        
        std::cout << "Image size: " << leftImg.cols << "x" << leftImg.rows << "\n\n";
        
        std::cout << "Middlebury calib: doffs=" << camParams.disparityOffset
                  << " px, ndisp=" << camParams.numDisparities
                  << ", baseline=" << (camParams.baseline * 1000.0) << " mm\n\n";

        // Run SGBM (full Middlebury ndisp when present; same default as evaluate/compare)
        stereo::StereoParams params;
        params.minDisparity = 0;
        params.maxDisparity = stereo::middleburyMatcherMaxDisparity(camParams.numDisparities, 128);
        if (camParams.numDisparities > stereo::kSgbmMaxDisparityRange) {
            std::cerr << "Warning: calib ndisp=" << camParams.numDisparities
                      << " exceeds GPU limit; using " << stereo::kSgbmMaxDisparityRange << "\n";
        }
        std::cout << "SGBM disparity search: 0 .. " << params.maxDisparity << "\n";
        params.blockSize = 5;
        params.P1 = 200;
        params.P2 = 1600;
        params.uniquenessRatio = 5;
        params.disp12MaxDiff = 3;
        params.speckleWindowSize = 0;
        params.speckleRange = 24;
        params.numDirections = 4;
        params.matchingCostMode = stereo::MatchingCostMode::Census;
        
        stereo::StereoSGBM sgbm(params);
        cv::Mat disparity;
        
        std::cout << "Computing disparity...\n";
        sgbm.compute(leftImg, rightImg, disparity);
        std::cout << "Done in " << sgbm.getLastComputeTimeMs() << " ms\n\n";
        
        // Analyze disparity values
        std::cout << "=== Disparity Analysis ===\n";
        std::cout << "Type: CV_" << (disparity.type() == CV_16SC1 ? "16SC1" : "unknown") << "\n";
        
        int64_t sum = 0;
        int validCount = 0;
        int zeroCount = 0;
        int negCount = 0;
        int16_t minVal = INT16_MAX, maxVal = INT16_MIN;
        
        for (int y = 0; y < disparity.rows; y++) {
            for (int x = 0; x < disparity.cols; x++) {
                int16_t d = disparity.at<int16_t>(y, x);
                if (d < 0) {
                    negCount++;
                } else if (d == 0) {
                    zeroCount++;
                } else {
                    validCount++;
                    sum += d;
                    minVal = std::min(minVal, d);
                    maxVal = std::max(maxVal, d);
                }
            }
        }
        
        int totalPixels = disparity.rows * disparity.cols;
        
        std::cout << "Total pixels: " << totalPixels << "\n";
        std::cout << "Valid (d > 0): " << validCount << " (" << (100.0 * validCount / totalPixels) << "%)\n";
        std::cout << "Zero (d = 0): " << zeroCount << " (" << (100.0 * zeroCount / totalPixels) << "%)\n";
        std::cout << "Invalid (d < 0): " << negCount << " (" << (100.0 * negCount / totalPixels) << "%)\n";
        
        if (validCount > 0) {
            double avgDisp = static_cast<double>(sum) / validCount / 16.0;  // Convert from 1/16 px
            std::cout << "Min disparity: " << (minVal / 16.0) << " px\n";
            std::cout << "Max disparity: " << (maxVal / 16.0) << " px\n";
            std::cout << "Avg disparity: " << avgDisp << " px\n";
            
            // Compute depth
            double denom = avgDisp + static_cast<double>(camParams.disparityOffset);
            double avgDepth = (denom > 0.0) ? (camParams.focalLength * camParams.baseline / denom) : 0.0;
            std::cout << "\nEstimated average depth: " << avgDepth << " m\n";
        }
        
        // Sample some specific pixels
        std::cout << "\n=== Sample Pixels (center region) ===\n";
        int cy = disparity.rows / 2;
        int cx = disparity.cols / 2;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int16_t d = disparity.at<int16_t>(cy + dy * 50, cx + dx * 50);
                std::cout << "(" << (cx + dx*50) << "," << (cy + dy*50) << "): ";
                if (d <= 0) {
                    std::cout << "invalid";
                } else {
                    std::cout << "d=" << (d/16.0) << " px";
                }
                std::cout << "  ";
            }
            std::cout << "\n";
        }
        
        // Save enhanced visualization
        cv::Mat dispVis;
        double minD = 0.0, maxD = 0.0;
        cv::minMaxLoc(disparity, &minD, &maxD, nullptr, nullptr, disparity > 0);
        if (maxD <= 0.0) {
            dispVis = cv::Mat::zeros(disparity.size(), CV_8U);
        } else {
            disparity.convertTo(dispVis, CV_8U, 255.0 / maxD);
            dispVis.setTo(0, disparity <= 0);
        }
        cv::imwrite("disparity_enhanced.png", dispVis);
        std::cout << "\nSaved enhanced disparity to disparity_enhanced.png\n";
        
        // Also save a pseudo-color version
        cv::Mat dispColor;
        cv::applyColorMap(dispVis, dispColor, cv::COLORMAP_TURBO);
        cv::imwrite("disparity_color.png", dispColor);
        std::cout << "Saved colorized disparity to disparity_color.png\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
