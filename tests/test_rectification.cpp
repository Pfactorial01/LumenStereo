/**
 * test_rectification.cpp - Rectification and calibration tests
 */

#include <gtest/gtest.h>
#include "stereo/calibration.h"
#include "stereo/rectification.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace stereo {
namespace test {

// ============================================================================
// CameraParams Tests
// ============================================================================

TEST(CameraParamsTest, DefaultInvalid) {
    CameraParams params;
    EXPECT_FALSE(params.isValid());  // Empty matrices
}

TEST(CameraParamsTest, ComputeDerivedValues) {
    CameraParams params;
    params.width = 640;
    params.height = 480;
    
    // Set up intrinsics
    params.K0 = cv::Mat::eye(3, 3, CV_64F);
    params.K0.at<double>(0, 0) = 500.0;  // fx
    params.K0.at<double>(1, 1) = 500.0;  // fy
    params.K0.at<double>(0, 2) = 320.0;  // cx
    params.K0.at<double>(1, 2) = 240.0;  // cy
    
    params.K1 = params.K0.clone();
    
    // Set up extrinsics
    params.R = cv::Mat::eye(3, 3, CV_64F);
    params.T = cv::Mat::zeros(3, 1, CV_64F);
    params.T.at<double>(0) = -0.1;  // 100mm baseline
    
    params.D0 = cv::Mat::zeros(5, 1, CV_64F);
    params.D1 = cv::Mat::zeros(5, 1, CV_64F);
    
    params.computeDerivedValues();
    
    EXPECT_NEAR(params.baseline, 0.1, 1e-6);
    EXPECT_NEAR(params.focalLength, 500.0, 1e-6);
    EXPECT_TRUE(params.isValid());
}

// ============================================================================
// Middlebury Calibration Tests
// ============================================================================

TEST(CalibrationLoadTest, MiddleburyFormat) {
    CameraParams params;
    
    // Try to load Middlebury calibration
    bool loaded = loadMiddleburyCalib("../dataset/Adirondack-perfect/calib.txt", params);
    
    if (loaded) {
        EXPECT_GT(params.width, 0);
        EXPECT_GT(params.height, 0);
        EXPECT_GT(params.focalLength, 0);
        EXPECT_GT(params.baseline, 0);
        
        std::cout << "Middlebury calibration:\n";
        std::cout << "  Resolution: " << params.width << "x" << params.height << "\n";
        std::cout << "  Focal length: " << params.focalLength << " px\n";
        std::cout << "  Baseline: " << params.baseline * 1000 << " mm\n";
    } else {
        GTEST_SKIP() << "Middlebury dataset not found (OK for CI)";
    }
}

TEST(CalibrationLoadTest, NonexistentFile) {
    CameraParams params;
    bool loaded = loadMiddleburyCalib("/nonexistent/path/calib.txt", params);
    EXPECT_FALSE(loaded);
}

// ============================================================================
// YAML Save/Load Tests
// ============================================================================

TEST(ConfigTest, StereoParamsSaveLoad) {
    StereoParams original;
    original.minDisparity = 10;
    original.maxDisparity = 200;
    original.blockSize = 7;
    original.P1 = 15;
    original.P2 = 60;
    
    std::string path = "/tmp/test_stereo_params.yaml";
    
    ASSERT_TRUE(saveStereoParams(path, original));
    
    StereoParams loaded;
    ASSERT_TRUE(loadStereoParams(path, loaded));
    
    EXPECT_EQ(loaded.minDisparity, original.minDisparity);
    EXPECT_EQ(loaded.maxDisparity, original.maxDisparity);
    EXPECT_EQ(loaded.blockSize, original.blockSize);
    EXPECT_EQ(loaded.P1, original.P1);
    EXPECT_EQ(loaded.P2, original.P2);
    
    // Cleanup
    std::remove(path.c_str());
}

TEST(ConfigTest, StereoParamsMatchingCostRoundtrip) {
    StereoParams original;
    original.matchingCostMode = MatchingCostMode::Census;
    std::string path = "/tmp/test_stereo_params_census.yaml";
    ASSERT_TRUE(saveStereoParams(path, original));
    StereoParams loaded;
    ASSERT_TRUE(loadStereoParams(path, loaded));
    EXPECT_EQ(loaded.matchingCostMode, MatchingCostMode::Census);
    std::remove(path.c_str());
}

TEST(ConfigTest, CameraParamsSaveLoad) {
    CameraParams original;
    original.width = 1920;
    original.height = 1080;
    original.K0 = cv::Mat::eye(3, 3, CV_64F);
    original.K0.at<double>(0, 0) = 1000.0;
    original.K0.at<double>(1, 1) = 1000.0;
    original.K1 = original.K0.clone();
    original.D0 = cv::Mat::zeros(5, 1, CV_64F);
    original.D1 = cv::Mat::zeros(5, 1, CV_64F);
    original.R = cv::Mat::eye(3, 3, CV_64F);
    original.T = cv::Mat::zeros(3, 1, CV_64F);
    original.T.at<double>(0) = -0.12;
    original.computeDerivedValues();
    
    std::string path = "/tmp/test_camera_params.yaml";
    
    ASSERT_TRUE(saveCameraParams(path, original));
    
    CameraParams loaded;
    ASSERT_TRUE(loadCameraParams(path, loaded));
    
    EXPECT_EQ(loaded.width, original.width);
    EXPECT_EQ(loaded.height, original.height);
    EXPECT_NEAR(loaded.baseline, original.baseline, 1e-6);
    
    // Cleanup
    std::remove(path.c_str());
}

// ============================================================================
// GPU Rectification Tests
// ============================================================================

TEST(RectificationTest, PreRectifiedPassthrough) {
    try {
        // Create params without rectification maps (simulates pre-rectified)
        CameraParams params;
        params.width = 640;
        params.height = 480;
        // map0x/map0y are empty
        
        StereoRectifier rectifier(params);
        EXPECT_TRUE(rectifier.isPreRectified());
        
        cv::Mat left(480, 640, CV_8UC3);
        cv::Mat right(480, 640, CV_8UC3);
        cv::randu(left, 0, 255);
        cv::randu(right, 0, 255);
        
        cv::Mat leftRect, rightRect;
        rectifier.rectify(left, right, leftRect, rightRect);
        
        // Should be identical (just copied)
        cv::Mat diff;
        cv::absdiff(left, leftRect, diff);
        EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(RectificationTest, GPUvsOpenCV) {
    try {
        int width = 640;
        int height = 480;
        
        // Create test image
        cv::Mat src(height, width, CV_8UC1);
        cv::randu(src, 0, 255);
        
        // Create identity map
        cv::Mat mapX(height, width, CV_32FC1);
        cv::Mat mapY(height, width, CV_32FC1);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mapX.at<float>(y, x) = static_cast<float>(x);
                mapY.at<float>(y, x) = static_cast<float>(y);
            }
        }
        
        cv::Mat dstCPU, dstGPU;
        cv::remap(src, dstCPU, mapX, mapY, cv::INTER_LINEAR);
        rectifyImageGPU(src, mapX, mapY, dstGPU);
        
        cv::Mat diff;
        cv::absdiff(dstCPU, dstGPU, diff);
        double maxDiff;
        cv::minMaxLoc(diff, nullptr, &maxDiff);
        
        // Should match exactly for identity map
        EXPECT_EQ(maxDiff, 0) << "Identity map should produce identical results";
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(RectificationTest, BilinearInterpolation) {
    try {
        int width = 100;
        int height = 100;
        
        // Create gradient image
        cv::Mat src(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                src.at<uchar>(y, x) = static_cast<uchar>(x * 255 / width);
            }
        }
        
        // Create map with 0.5 pixel offset
        cv::Mat mapX(height, width, CV_32FC1);
        cv::Mat mapY(height, width, CV_32FC1);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mapX.at<float>(y, x) = static_cast<float>(x) + 0.5f;
                mapY.at<float>(y, x) = static_cast<float>(y) + 0.5f;
            }
        }
        
        cv::Mat dstCPU, dstGPU;
        cv::remap(src, dstCPU, mapX, mapY, cv::INTER_LINEAR);
        rectifyImageGPU(src, mapX, mapY, dstGPU);
        
        cv::Mat diff;
        cv::absdiff(dstCPU, dstGPU, diff);
        double maxDiff;
        cv::minMaxLoc(diff, nullptr, &maxDiff);
        
        // Allow small rounding differences
        EXPECT_LE(maxDiff, 1) << "Bilinear interpolation should match within 1 pixel value";
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

// ============================================================================
// Checkerboard Detection Tests
// ============================================================================

TEST(CalibrationTest, CheckerboardDetection) {
    // Create synthetic checkerboard image
    int squareSize = 50;
    cv::Size boardSize(9, 6);
    
    int width = (boardSize.width + 2) * squareSize;
    int height = (boardSize.height + 2) * squareSize;
    
    cv::Mat board(height, width, CV_8UC1, cv::Scalar(255));
    
    // Draw checkerboard
    for (int y = 0; y < boardSize.height + 1; y++) {
        for (int x = 0; x < boardSize.width + 1; x++) {
            if ((x + y) % 2 == 0) continue;
            
            int x0 = (x + 1) * squareSize;
            int y0 = (y + 1) * squareSize;
            cv::rectangle(board, cv::Point(x0, y0), 
                         cv::Point(x0 + squareSize, y0 + squareSize),
                         cv::Scalar(0), -1);
        }
    }
    
    // Try to detect corners
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(board, boardSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    
    EXPECT_TRUE(found) << "Should detect checkerboard in synthetic image";
    
    if (found) {
        EXPECT_EQ(corners.size(), static_cast<size_t>(boardSize.width * boardSize.height));
    }
}

} // namespace test
} // namespace stereo
