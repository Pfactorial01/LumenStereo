/**
 * benchmark.cpp - Command-line stereo benchmarking tool
 * 
 * Runs SGBM on stereo pairs and outputs timing + saves results.
 * No GUI required.
 * 
 * Usage:
 *   ./benchmark --middlebury <dataset_path>
 */

#include "stereo/sgbm_gpu.h"
#include "stereo/depth_map.h"
#include "stereo/point_cloud.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"
#include "stereo/sgbm_limits.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--middlebury") {
        std::cout << "Usage: ./benchmark --middlebury <dataset_path>\n";
        return 1;
    }
    
    try {
        // Print GPU info
        stereo::printDeviceInfo();
        std::cout << "\n";
        
        std::string datasetPath = argv[2];
        
        // Load calibration
        stereo::CameraParams camParams;
        if (!stereo::loadMiddleburyCalib(datasetPath + "/calib.txt", camParams)) {
            std::cerr << "Failed to load calibration\n";
            return 1;
        }
        
        // Load images
        std::cout << "Loading images...\n";
        cv::Mat leftImg = cv::imread(datasetPath + "/im0.png");
        cv::Mat rightImg = cv::imread(datasetPath + "/im1.png");
        
        if (leftImg.empty() || rightImg.empty()) {
            std::cerr << "Failed to load images. Make sure to extract zip files first.\n";
            return 1;
        }
        
        std::cout << "Image size: " << leftImg.cols << "x" << leftImg.rows << "\n";
        std::cout << "Image type: " << (leftImg.channels() == 3 ? "BGR" : "Gray") << "\n\n";
        
        // Test different disparity ranges
        std::vector<int> dispRanges = {64, 128};
        cv::Mat lastDisparity;
        int lastMaxDisp = 0;
        
        for (int maxDisp : dispRanges) {
            std::cout << "=== Testing maxDisparity=" << maxDisp << " ===\n";
            
            // Estimate memory requirement (uint8 cost volume + uint16 aggregated + disparity buffers)
            size_t pixels = static_cast<size_t>(leftImg.cols) * leftImg.rows;
            size_t costBytes = pixels * static_cast<size_t>(maxDisp) * 1;
            size_t aggBytes = pixels * static_cast<size_t>(maxDisp) * 2;
            size_t dispBytes = pixels * sizeof(int16_t) * 3;  // disp + dispR + temp (approx)
            size_t totalNeeded = costBytes + aggBytes + dispBytes;
            std::cout << "Estimated GPU memory needed: " << (totalNeeded / 1e9) << " GB\n";
            
            try {
                stereo::StereoParams params;
                params.maxDisparity = maxDisp;
                params.blockSize = 5;
                params.autoComputePenalties();
                
                stereo::StereoSGBM sgbm(params);
                
                cv::Mat disparity;
                
                // Warm-up run
                std::cout << "Warm-up run...\n";
                sgbm.compute(leftImg, rightImg, disparity);
                
                // Benchmark run
                const int numRuns = 3;
                float totalTime = 0;
                
                std::cout << "Benchmark (" << numRuns << " runs)...\n";
                for (int i = 0; i < numRuns; i++) {
                    sgbm.compute(leftImg, rightImg, disparity);
                    float t = sgbm.getLastComputeTimeMs();
                    totalTime += t;
                    std::cout << "  Run " << (i+1) << ": " << t << " ms\n";
                }
                
                float avgTime = totalTime / numRuns;
                float fps = 1000.0f / avgTime;
                
                std::cout << "Average time: " << avgTime << " ms (" << fps << " FPS)\n";
                std::cout << "  Cost computation: " << sgbm.getCostComputeTimeMs() << " ms\n";
                std::cout << "  Aggregation:      " << sgbm.getAggregationTimeMs() << " ms\n";
                std::cout << "  Post-processing:  " << sgbm.getPostProcessTimeMs() << " ms\n";
                std::cout << "\n";
                
                // Save for later output
                lastDisparity = disparity.clone();
                lastMaxDisp = maxDisp;
                
            } catch (const stereo::CudaException& e) {
                std::cerr << "CUDA error (likely OOM): " << e.what() << "\n";
                std::cerr << "Skipping this configuration.\n\n";
            }
        }
        
        // Save outputs from the last successful run
        if (!lastDisparity.empty()) {
            std::cout << "=== Saving outputs ===\n";
            
            // Normalize for visualization
            cv::Mat dispVis;
            lastDisparity.convertTo(dispVis, CV_8U, 255.0 / (lastMaxDisp * 16));
            cv::imwrite("disparity_output.png", dispVis);
            std::cout << "Saved disparity to disparity_output.png\n";
            
            // Convert to depth and colorize
            cv::Mat depth, depthColor;
            stereo::disparityToDepth(lastDisparity, depth, camParams.focalLength, camParams.baseline,
                                     static_cast<double>(camParams.disparityOffset));
            stereo::colorizeDepth(depth, depthColor, 0.5f, 20.0f);
            cv::imwrite("depth_color_output.png", depthColor);
            std::cout << "Saved colorized depth to depth_color_output.png\n";
            
            // Generate point cloud
            std::vector<stereo::Point3D> points = stereo::depthToPointCloud(depth, leftImg, camParams);
            std::cout << "Generated " << points.size() << " 3D points\n";
            
            stereo::exportPLY(points, "point_cloud_output.ply", false);
            std::cout << "Saved point cloud to point_cloud_output.ply\n";
        }
        
        std::cout << "\nDone!\n";
        
    } catch (const stereo::CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
