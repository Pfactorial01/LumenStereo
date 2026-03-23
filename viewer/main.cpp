/**
 * main.cpp - LumenStereo Interactive Viewer Application
 * 
 * Usage:
 *   ./stereo_viewer --middlebury <dataset_path>
 *   ./stereo_viewer <left_image> <right_image> [calib_file]
 */

#include "stereo_viewer.h"
#include "stereo/sgbm_gpu.h"
#include "stereo/depth_map.h"
#include "stereo/point_cloud.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <memory>

void printUsage(const char* programName) {
    std::cout << "LumenStereo - Real-Time GPU Stereo Depth Estimation\n\n"
              << "Usage:\n"
              << "  " << programName << " --middlebury <dataset_path>\n"
              << "  " << programName << " <left_image> <right_image> [calib_file]\n"
              << "\nOptions:\n"
              << "  --help          Show this help message\n"
              << "  --middlebury    Load Middlebury stereo dataset\n"
              << "\nControls:\n"
              << "  - Use sliders to adjust SGBM parameters\n"
              << "  - Click 'Recompute' to apply changes\n"
              << "  - Click on disparity image to probe depth\n"
              << "  - Press ESC or close window to exit\n";
}

class StereoApp {
public:
    bool loadMiddlebury(const std::string& path) {
        // Load calibration
        if (!stereo::loadMiddleburyCalib(path + "/calib.txt", camParams_)) {
            std::cerr << "Failed to load calibration\n";
            return false;
        }
        
        // Load images
        leftImg_ = cv::imread(path + "/im0.png");
        rightImg_ = cv::imread(path + "/im1.png");
        
        if (leftImg_.empty() || rightImg_.empty()) {
            std::cerr << "Failed to load images. Extract .zip files first.\n";
            return false;
        }
        
        std::cout << "Loaded: " << leftImg_.cols << "x" << leftImg_.rows << std::endl;
        std::cout.flush();
        return true;
    }
    
    bool loadImages(const std::string& leftPath, const std::string& rightPath,
                    const std::string& calibPath = "") {
        leftImg_ = cv::imread(leftPath);
        rightImg_ = cv::imread(rightPath);
        
        if (leftImg_.empty() || rightImg_.empty()) {
            std::cerr << "Failed to load images\n";
            return false;
        }
        
        if (!calibPath.empty()) {
            stereo::loadCameraParams(calibPath, camParams_);
        } else {
            // Use defaults
            camParams_.width = leftImg_.cols;
            camParams_.height = leftImg_.rows;
            camParams_.focalLength = 1000.0;
            camParams_.baseline = 0.1;
        }
        
        return true;
    }
    
    void computeDisparity(const stereo::StereoParams& params) {
        if (leftImg_.empty()) return;
        
        std::cout << "Computing disparity (maxDisp=" << params.maxDisparity 
                  << ", blockSize=" << params.blockSize << ")...\n";
        
        try {
            // Create SGBM with current parameters
            stereo::StereoSGBM sgbm(params);
            
            // Compute
            sgbm.compute(leftImg_, rightImg_, disparity_);
            
            // Store timing
            totalTimeMs_ = sgbm.getLastComputeTimeMs();
            costTimeMs_ = sgbm.getCostComputeTimeMs();
            aggTimeMs_ = sgbm.getAggregationTimeMs();
            postTimeMs_ = sgbm.getPostProcessTimeMs();
            
            std::cout << "Done in " << totalTimeMs_ << " ms\n";
            
            // Convert to depth and colorize
            stereo::disparityToDepth(disparity_, depth_, camParams_.focalLength, camParams_.baseline,
                                     static_cast<double>(camParams_.disparityOffset));
            
            // Data-dependent depth range: manual min/max over valid depth (depth > 0)
            float minDepth = 1e6f, maxDepth = 0.f;
            for (int y = 0; y < depth_.rows; y++) {
                const float* row = depth_.ptr<float>(y);
                for (int x = 0; x < depth_.cols; x++) {
                    float d = row[x];
                    if (d > 0) {
                        if (d < minDepth) minDepth = d;
                        if (d > maxDepth) maxDepth = d;
                    }
                }
            }
            if (maxDepth <= minDepth) {
                minDepth = 0.5f;
                maxDepth = 30.0f;
            } else {
                minDepth = std::max(0.1f, minDepth);
                maxDepth = std::max(maxDepth, minDepth + 0.1f);
            }
            stereo::colorizeDepth(depth_, depthColor_, minDepth, maxDepth);
            depthColor_.setTo(cv::Scalar(0, 0, 0), depth_ <= 0);
            
            cudaAvailable_ = true;
        } catch (const stereo::CudaException& e) {
            std::cerr << "CUDA Error: " << e.what() << "\n";
            std::cerr << "GPU computation unavailable. Check your NVIDIA driver.\n";
            cudaAvailable_ = false;
            
            // Create placeholder images
            disparity_ = cv::Mat::zeros(leftImg_.size(), CV_16SC1);
            depthColor_ = cv::Mat::zeros(leftImg_.size(), CV_8UC3);
            cv::putText(depthColor_, "CUDA Error - GPU unavailable", 
                       cv::Point(50, depthColor_.rows/2), cv::FONT_HERSHEY_SIMPLEX, 
                       2.0, cv::Scalar(0, 0, 255), 3);
        }
    }
    
    void run() {
        // Initialize viewer
        viewer_ = std::make_unique<stereo::StereoViewer>("LumenStereo - Interactive Viewer");
        
        if (!viewer_->init(1600, 900)) {
            std::cerr << "Failed to initialize viewer" << std::endl;
            return;
        }
        
        // Print GPU info after OpenGL context is created
        stereo::printDeviceInfo();
        std::cout << std::endl;
        
        // Set camera parameters
        viewer_->setCameraParams(camParams_);
        
        // Set up recompute callback
        viewer_->setRecomputeCallback([this]() {
            computeDisparity(viewer_->getParams());
            updateViewer();
        });
        
        // Initial computation
        computeDisparity(viewer_->getParams());
        updateViewer();
        
        // Main loop
        while (viewer_->render()) {
            // If user changed params and wants live update, could recompute here
            // For now, require explicit "Recompute" button click
        }
        
        viewer_->shutdown();
    }
    
private:
    void updateViewer() {
        viewer_->setImages(leftImg_, rightImg_, disparity_, depthColor_);
        viewer_->setDisparityRaw(disparity_);
        viewer_->setTiming(totalTimeMs_, costTimeMs_, aggTimeMs_, postTimeMs_);
    }
    
    std::unique_ptr<stereo::StereoViewer> viewer_;
    
    cv::Mat leftImg_, rightImg_;
    cv::Mat disparity_, depth_, depthColor_;
    stereo::CameraParams camParams_;
    
    float totalTimeMs_ = 0;
    float costTimeMs_ = 0;
    float aggTimeMs_ = 0;
    float postTimeMs_ = 0;
    bool cudaAvailable_ = true;
};

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }
    
    try {
        StereoApp app;
        
        // Parse arguments
        if (std::string(argv[1]) == "--middlebury") {
            if (argc < 3) {
                std::cerr << "Error: Middlebury mode requires dataset path\n";
                return 1;
            }
            
            if (!app.loadMiddlebury(argv[2])) {
                return 1;
            }
        } else {
            if (argc < 3) {
                std::cerr << "Error: Need left and right image paths\n";
                printUsage(argv[0]);
                return 1;
            }
            
            std::string calibPath = argc > 3 ? argv[3] : "";
            if (!app.loadImages(argv[1], argv[2], calibPath)) {
                return 1;
            }
        }
        
        // Run interactive viewer
        app.run();
        
    } catch (const stereo::CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
