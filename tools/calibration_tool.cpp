/**
 * calibration_tool.cpp - Command-line stereo calibration tool
 * 
 * Usage:
 *   # Calibrate from image directory
 *   ./calibration_tool calibrate --images <dir> --board 9x6 --square 25.0 --output calibration.yaml
 * 
 *   # Test calibration by rectifying images
 *   ./calibration_tool rectify --calib calibration.yaml --left img_left.png --right img_right.png
 * 
 *   # Generate checkerboard PDF
 *   ./calibration_tool generate-board --size 9x6 --square 25 --output checkerboard.png
 * 
 *   # Test with Middlebury dataset (already rectified)
 *   ./calibration_tool test-middlebury --path dataset/Adirondack-perfect
 */

#include "stereo/calibration.h"
#include "stereo/rectification.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <chrono>

namespace fs = std::filesystem;

void printUsage() {
    std::cout << R"(
Stereo Calibration Tool
=======================

Commands:
  calibrate         Calibrate stereo camera from image pairs
  rectify           Apply calibration to rectify a stereo pair
  benchmark-rectify Benchmark GPU vs CPU rectification
  generate-board    Generate a checkerboard pattern image
  test-middlebury   Test loading Middlebury calibration

Usage:
  ./calibration_tool calibrate --images <dir> --board 9x6 --square 25.0 --output calib.yaml
    --images   Directory containing left*.png and right*.png pairs
    --board    Checkerboard inner corners (cols x rows)
    --square   Square size in millimeters
    --output   Output calibration file (YAML)

  ./calibration_tool rectify --calib calib.yaml --left left.png --right right.png
    --calib    Calibration file
    --left     Left input image
    --right    Right input image
    --output   Output directory (default: current)

  ./calibration_tool generate-board --size 9x6 --square 25 --output board.png
    --size     Board size (cols x rows)
    --square   Square size in pixels
    --output   Output image file

  ./calibration_tool test-middlebury --path <dataset_path>
    --path     Path to Middlebury dataset

)";
}

/**
 * Parse board size from string like "9x6"
 */
cv::Size parseBoardSize(const std::string& str) {
    std::regex re("(\\d+)x(\\d+)");
    std::smatch match;
    if (std::regex_match(str, match, re)) {
        return cv::Size(std::stoi(match[1]), std::stoi(match[2]));
    }
    throw stereo::StereoException("Invalid board size format. Use WxH (e.g., 9x6)");
}

/**
 * Find stereo image pairs in a directory
 * Looks for patterns like: left_01.png/right_01.png or left1.png/right1.png
 */
std::vector<std::pair<std::string, std::string>> findImagePairs(const std::string& dir) {
    std::vector<std::pair<std::string, std::string>> pairs;
    
    std::vector<std::string> leftImages, rightImages;
    
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string filename = entry.path().filename().string();
        std::string fullpath = entry.path().string();
        
        // Check for left images
        if (filename.find("left") != std::string::npos || 
            filename.find("Left") != std::string::npos ||
            filename.find("cam0") != std::string::npos ||
            filename.find("im0") != std::string::npos) {
            leftImages.push_back(fullpath);
        }
        // Check for right images
        else if (filename.find("right") != std::string::npos || 
                 filename.find("Right") != std::string::npos ||
                 filename.find("cam1") != std::string::npos ||
                 filename.find("im1") != std::string::npos) {
            rightImages.push_back(fullpath);
        }
    }
    
    // Sort both lists
    std::sort(leftImages.begin(), leftImages.end());
    std::sort(rightImages.begin(), rightImages.end());
    
    // Pair them up (assuming same count and matching order)
    size_t count = std::min(leftImages.size(), rightImages.size());
    for (size_t i = 0; i < count; i++) {
        pairs.push_back({leftImages[i], rightImages[i]});
    }
    
    return pairs;
}

/**
 * Calibrate command
 */
int cmdCalibrate(int argc, char** argv) {
    std::string imagesDir;
    std::string boardSizeStr = "9x6";
    float squareSize = 25.0f;
    std::string outputPath = "calibration.yaml";
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--images" && i + 1 < argc) {
            imagesDir = argv[++i];
        } else if (arg == "--board" && i + 1 < argc) {
            boardSizeStr = argv[++i];
        } else if (arg == "--square" && i + 1 < argc) {
            squareSize = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        }
    }
    
    if (imagesDir.empty()) {
        std::cerr << "Error: --images directory required\n";
        return 1;
    }
    
    cv::Size boardSize = parseBoardSize(boardSizeStr);
    
    std::cout << "Stereo Calibration\n";
    std::cout << "==================\n";
    std::cout << "Images directory: " << imagesDir << "\n";
    std::cout << "Board size: " << boardSize << "\n";
    std::cout << "Square size: " << squareSize << " mm\n";
    std::cout << "Output: " << outputPath << "\n\n";
    
    // Find image pairs
    auto pairs = findImagePairs(imagesDir);
    
    if (pairs.empty()) {
        std::cerr << "No image pairs found in " << imagesDir << "\n";
        std::cerr << "Expected files named left*.png and right*.png\n";
        return 1;
    }
    
    std::cout << "Found " << pairs.size() << " image pairs\n\n";
    
    // Create calibrator
    stereo::StereoCalibrator calibrator(boardSize, squareSize);
    
    // Process each pair
    for (size_t i = 0; i < pairs.size(); i++) {
        std::cout << "Processing pair " << (i+1) << "/" << pairs.size() << ": "
                  << fs::path(pairs[i].first).filename() << " + "
                  << fs::path(pairs[i].second).filename() << "\n";
        
        cv::Mat left = cv::imread(pairs[i].first);
        cv::Mat right = cv::imread(pairs[i].second);
        
        if (left.empty() || right.empty()) {
            std::cerr << "  Failed to load images!\n";
            continue;
        }
        
        bool success = calibrator.addImagePair(left, right);
        if (!success) {
            std::cout << "  Checkerboard not detected\n";
        }
    }
    
    std::cout << "\nValid pairs with checkerboard: " << calibrator.numPairs() << "\n";
    
    if (calibrator.numPairs() < 5) {
        std::cerr << "Error: Need at least 5 valid pairs for calibration\n";
        return 1;
    }
    
    // Run calibration
    try {
        stereo::CameraParams params = calibrator.calibrate();
        
        // Save results
        if (stereo::saveCameraParams(outputPath, params)) {
            std::cout << "\nCalibration saved to " << outputPath << "\n";
        } else {
            std::cerr << "Failed to save calibration\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Calibration failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

/**
 * Rectify command
 */
int cmdRectify(int argc, char** argv) {
    std::string calibPath;
    std::string leftPath;
    std::string rightPath;
    std::string outputDir = ".";
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--calib" && i + 1 < argc) {
            calibPath = argv[++i];
        } else if (arg == "--left" && i + 1 < argc) {
            leftPath = argv[++i];
        } else if (arg == "--right" && i + 1 < argc) {
            rightPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[++i];
        }
    }
    
    if (calibPath.empty() || leftPath.empty() || rightPath.empty()) {
        std::cerr << "Error: --calib, --left, and --right are required\n";
        return 1;
    }
    
    // Load calibration
    stereo::CameraParams params;
    if (!stereo::loadCameraParams(calibPath, params)) {
        std::cerr << "Failed to load calibration from " << calibPath << "\n";
        return 1;
    }
    
    // Ensure rectification maps are computed
    if (params.map0x.empty()) {
        std::cout << "Computing rectification maps...\n";
        stereo::computeRectification(params);
    }
    
    // Load images
    cv::Mat left = cv::imread(leftPath);
    cv::Mat right = cv::imread(rightPath);
    
    if (left.empty() || right.empty()) {
        std::cerr << "Failed to load images\n";
        return 1;
    }
    
    // Rectify
    cv::Mat leftRect, rightRect;
    stereo::rectifyImages(params, left, right, leftRect, rightRect);
    
    // Save results
    std::string outLeft = outputDir + "/left_rectified.png";
    std::string outRight = outputDir + "/right_rectified.png";
    std::string outVis = outputDir + "/rectification_check.png";
    
    cv::imwrite(outLeft, leftRect);
    cv::imwrite(outRight, rightRect);
    
    // Create visualization
    cv::Mat vis = stereo::visualizeRectification(leftRect, rightRect);
    cv::imwrite(outVis, vis);
    
    std::cout << "Saved rectified images:\n";
    std::cout << "  " << outLeft << "\n";
    std::cout << "  " << outRight << "\n";
    std::cout << "  " << outVis << " (with epipolar lines)\n";
    
    return 0;
}

/**
 * Generate checkerboard command
 */
int cmdGenerateBoard(int argc, char** argv) {
    std::string sizeStr = "9x6";
    int squareSize = 50;  // pixels
    std::string outputPath = "checkerboard.png";
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            sizeStr = argv[++i];
        } else if (arg == "--square" && i + 1 < argc) {
            squareSize = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        }
    }
    
    cv::Size boardSize = parseBoardSize(sizeStr);
    
    // Create checkerboard image
    // Board size is inner corners, so we need one more row/col of squares
    int width = (boardSize.width + 1) * squareSize;
    int height = (boardSize.height + 1) * squareSize;
    
    cv::Mat board(height, width, CV_8UC1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int squareX = x / squareSize;
            int squareY = y / squareSize;
            bool isWhite = (squareX + squareY) % 2 == 0;
            board.at<uchar>(y, x) = isWhite ? 255 : 0;
        }
    }
    
    // Add white border for printing
    int border = squareSize;
    cv::Mat withBorder;
    cv::copyMakeBorder(board, withBorder, border, border, border, border,
                       cv::BORDER_CONSTANT, cv::Scalar(255));
    
    cv::imwrite(outputPath, withBorder);
    
    std::cout << "Generated checkerboard:\n";
    std::cout << "  Inner corners: " << boardSize << "\n";
    std::cout << "  Square size: " << squareSize << " pixels\n";
    std::cout << "  Image size: " << withBorder.cols << "x" << withBorder.rows << "\n";
    std::cout << "  Saved to: " << outputPath << "\n\n";
    std::cout << "Print this at a known size and measure the actual square size.\n";
    std::cout << "Use that measurement (in mm) for the --square parameter.\n";
    
    return 0;
}

/**
 * Test Middlebury command
 */
int cmdTestMiddlebury(int argc, char** argv) {
    std::string datasetPath;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            datasetPath = argv[++i];
        }
    }
    
    if (datasetPath.empty()) {
        std::cerr << "Error: --path required\n";
        return 1;
    }
    
    std::cout << "Testing Middlebury dataset: " << datasetPath << "\n\n";
    
    // Load Middlebury calibration
    stereo::CameraParams params;
    if (!stereo::loadMiddleburyCalib(datasetPath + "/calib.txt", params)) {
        std::cerr << "Failed to load Middlebury calibration\n";
        return 1;
    }
    
    std::cout << "Calibration loaded successfully!\n";
    std::cout << "  Resolution: " << params.width << "x" << params.height << "\n";
    std::cout << "  Focal length: " << params.focalLength << " px\n";
    std::cout << "  Baseline: " << params.baseline * 1000.0 << " mm\n";
    
    // Note: Middlebury images are already rectified, so we don't need
    // to compute rectification or apply it
    std::cout << "\nNote: Middlebury images are pre-rectified.\n";
    std::cout << "No additional rectification needed.\n";
    
    return 0;
}

/**
 * Benchmark rectification command
 */
int cmdBenchmarkRectify(int argc, char** argv) {
    int width = 1920;
    int height = 1080;
    int iterations = 100;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "Benchmarking GPU vs CPU rectification\n";
    std::cout << "Image size: " << width << "x" << height << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";
    
    // Create synthetic test image (grayscale for simpler debugging)
    cv::Mat src(height, width, CV_8UC1);
    cv::randu(src, 0, 255);
    
    // Create remap with 0.5 pixel shift (tests bilinear interpolation)
    cv::Mat mapX(height, width, CV_32FC1);
    cv::Mat mapY(height, width, CV_32FC1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            mapX.at<float>(y, x) = static_cast<float>(x) + 0.5f;
            mapY.at<float>(y, x) = static_cast<float>(y) + 0.5f;
        }
    }
    
    cv::Mat dstCPU, dstGPU;
    
    // Warm up
    cv::remap(src, dstCPU, mapX, mapY, cv::INTER_LINEAR);
    stereo::rectifyImageGPU(src, mapX, mapY, dstGPU);
    
    // CPU benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cv::remap(src, dstCPU, mapX, mapY, cv::INTER_LINEAR);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count() / iterations;
    
    // GPU benchmark
    auto gpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        stereo::rectifyImageGPU(src, mapX, mapY, dstGPU);
    }
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count() / iterations;
    
    std::cout << "Results:\n";
    std::cout << "  CPU (OpenCV cv::remap): " << cpuMs << " ms\n";
    std::cout << "  GPU (CUDA kernel):      " << gpuMs << " ms\n";
    std::cout << "  Speedup:                " << (cpuMs / gpuMs) << "x\n";
    std::cout << "  CPU FPS:                " << (1000.0 / cpuMs) << "\n";
    std::cout << "  GPU FPS:                " << (1000.0 / gpuMs) << "\n";
    
    // Verify correctness (compare outputs)
    cv::Mat diff;
    cv::absdiff(dstCPU, dstGPU, diff);
    double minDiff, maxDiff;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(diff, &minDiff, &maxDiff, &minLoc, &maxLoc);
    std::cout << "\nMax pixel difference: " << maxDiff << " at " << maxLoc << " (should be < 2)\n";
    
    // Show specific values at max diff location
    if (dstCPU.channels() == 1) {
        std::cout << "  CPU value: " << (int)dstCPU.at<uchar>(maxLoc) << "\n";
        std::cout << "  GPU value: " << (int)dstGPU.at<uchar>(maxLoc) << "\n";
        std::cout << "  Src value: " << (int)src.at<uchar>(maxLoc) << "\n";
        std::cout << "  MapX at loc: " << mapX.at<float>(maxLoc) << "\n";
        std::cout << "  MapY at loc: " << mapY.at<float>(maxLoc) << "\n";
    }
    
    // Count bad pixels
    int badPixels = cv::countNonZero(diff > 2);
    std::cout << "Bad pixels (diff > 2): " << badPixels << " (" 
              << (100.0 * badPixels / (width * height)) << "%)\n";
    
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string command = argv[1];
    
    try {
        if (command == "calibrate") {
            return cmdCalibrate(argc, argv);
        } else if (command == "rectify") {
            return cmdRectify(argc, argv);
        } else if (command == "benchmark-rectify") {
            return cmdBenchmarkRectify(argc, argv);
        } else if (command == "generate-board") {
            return cmdGenerateBoard(argc, argv);
        } else if (command == "test-middlebury") {
            return cmdTestMiddlebury(argc, argv);
        } else if (command == "--help" || command == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown command: " << command << "\n";
            printUsage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
