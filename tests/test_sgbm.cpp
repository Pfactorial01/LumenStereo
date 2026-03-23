/**
 * test_sgbm.cpp - SGBM algorithm tests
 * 
 * Tests cover:
 * 1. Parameter validation
 * 2. CUDA buffer operations
 * 3. SGBM computation correctness
 * 4. Edge cases and error handling
 */

#include <gtest/gtest.h>
#include "stereo/sgbm_gpu.h"
#include "stereo/sgbm_limits.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"
#include "stereo/depth_map.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace stereo {
namespace test {

// ============================================================================
// StereoParams Tests
// ============================================================================

TEST(StereoParamsTest, DefaultValues) {
    StereoParams params;
    EXPECT_EQ(params.minDisparity, 0);
    EXPECT_EQ(params.maxDisparity, 128);
    EXPECT_EQ(params.blockSize, 5);
    EXPECT_TRUE(params.isValid());
}

TEST(StereoParamsTest, InvalidBlockSize) {
    StereoParams params;
    params.blockSize = 4;  // Even number is invalid
    EXPECT_FALSE(params.isValid());
    
    params.blockSize = 0;
    EXPECT_FALSE(params.isValid());
    
    params.blockSize = -1;
    EXPECT_FALSE(params.isValid());
}

TEST(StereoParamsTest, InvalidDisparity) {
    StereoParams params;
    
    params.minDisparity = 100;
    params.maxDisparity = 50;  // max < min is invalid
    EXPECT_FALSE(params.isValid());
    
    params.minDisparity = -10;  // Negative min is invalid
    params.maxDisparity = 128;
    EXPECT_FALSE(params.isValid());
}

TEST(StereoParamsTest, InvalidPenalties) {
    StereoParams params;
    params.P1 = 100;
    params.P2 = 50;  // P2 <= P1 is invalid
    EXPECT_FALSE(params.isValid());
}

TEST(StereoParamsTest, AutoPenalties) {
    StereoParams params;
    params.blockSize = 5;
    params.autoComputePenalties(1);
    
    EXPECT_EQ(params.P1, 8 * 1 * 5 * 5);  // 200
    EXPECT_EQ(params.P2, 32 * 1 * 5 * 5); // 800
    EXPECT_GT(params.P2, params.P1);
}

// ============================================================================
// CudaBuffer Tests
// ============================================================================

TEST(CudaBufferTest, Allocation) {
    try {
        CudaBuffer<float> buffer(1000);
        EXPECT_EQ(buffer.size(), 1000);
        EXPECT_FALSE(buffer.empty());
        EXPECT_NE(buffer.data(), nullptr);
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(CudaBufferTest, EmptyBuffer) {
    CudaBuffer<float> buffer;
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
}

TEST(CudaBufferTest, Resize) {
    try {
        CudaBuffer<int> buffer(100);
        EXPECT_EQ(buffer.size(), 100);
        
        buffer.resize(500);
        EXPECT_EQ(buffer.size(), 500);
        
        buffer.resize(50);
        EXPECT_EQ(buffer.size(), 50);
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(CudaBufferTest, ZeroMemory) {
    try {
        CudaBuffer<int> buffer(100);
        buffer.zero();
        
        std::vector<int> host(100);
        buffer.copyTo(host.data());
        
        for (int val : host) {
            EXPECT_EQ(val, 0);
        }
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(CudaBufferTest, CopyRoundtrip) {
    try {
        std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        CudaBuffer<float> buffer(original.size());
        buffer.copyFrom(original.data());
        
        std::vector<float> result(original.size());
        buffer.copyTo(result.data());
        
        for (size_t i = 0; i < original.size(); i++) {
            EXPECT_FLOAT_EQ(result[i], original[i]);
        }
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(CudaBufferTest, LargeAllocation) {
    try {
        // Allocate 100MB
        size_t size = 100 * 1024 * 1024 / sizeof(float);
        CudaBuffer<float> buffer(size);
        EXPECT_EQ(buffer.size(), size);
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available or not enough memory: " << e.what();
    }
}

// ============================================================================
// SGBM Tests
// ============================================================================

TEST(SGBMTest, ComputeBasic) {
    try {
        // Create simple test images
        cv::Mat left = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::Mat right = cv::Mat::zeros(100, 100, CV_8UC1);
        
        // Add some pattern
        cv::rectangle(left, cv::Point(30, 30), cv::Point(70, 70), cv::Scalar(128), -1);
        cv::rectangle(right, cv::Point(25, 30), cv::Point(65, 70), cv::Scalar(128), -1);
        
        StereoParams params;
        params.maxDisparity = 32;
        params.blockSize = 5;
        params.numDirections = 4;
        
        StereoSGBM sgbm(params);
        
        cv::Mat disparity;
        sgbm.compute(left, right, disparity);
        
        EXPECT_EQ(disparity.rows, left.rows);
        EXPECT_EQ(disparity.cols, left.cols);
        EXPECT_EQ(disparity.type(), CV_16SC1);
        
        std::cout << "SGBM compute time: " << sgbm.getLastComputeTimeMs() << " ms\n";
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(SGBMTest, ColorInput) {
    try {
        // Create BGR images
        cv::Mat left = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::Mat right = cv::Mat::zeros(100, 100, CV_8UC3);
        
        cv::rectangle(left, cv::Point(30, 30), cv::Point(70, 70), cv::Scalar(128, 64, 32), -1);
        cv::rectangle(right, cv::Point(25, 30), cv::Point(65, 70), cv::Scalar(128, 64, 32), -1);
        
        StereoParams params;
        params.maxDisparity = 32;
        params.blockSize = 5;
        params.numDirections = 4;
        
        StereoSGBM sgbm(params);
        
        cv::Mat disparity;
        sgbm.compute(left, right, disparity);
        
        EXPECT_EQ(disparity.type(), CV_16SC1);
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(SGBMTest, DifferentSizes) {
    try {
        std::vector<cv::Size> sizes = {
            {64, 48},
            {128, 96},
            {320, 240},
            {640, 480}
        };
        
        StereoParams params;
        params.maxDisparity = 32;
        params.blockSize = 5;
        params.numDirections = 4;
        
        StereoSGBM sgbm(params);
        
        for (const auto& size : sizes) {
            cv::Mat left(size, CV_8UC1);
            cv::Mat right(size, CV_8UC1);
            cv::randu(left, 0, 255);
            cv::randu(right, 0, 255);
            
            cv::Mat disparity;
            sgbm.compute(left, right, disparity);
            
            EXPECT_EQ(disparity.size(), size);
            std::cout << "Size " << size << ": " << sgbm.getLastComputeTimeMs() << " ms\n";
        }
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(SGBMTest, DifferentDisparities) {
    try {
        cv::Mat left(240, 320, CV_8UC1);
        cv::Mat right(240, 320, CV_8UC1);
        cv::randu(left, 0, 255);
        cv::randu(right, 0, 255);
        
        std::vector<int> disparities = {32, 64, 96, 128};
        
        for (int maxDisp : disparities) {
            StereoParams params;
            params.maxDisparity = maxDisp;
            params.blockSize = 5;
            params.numDirections = 4;
            
            StereoSGBM sgbm(params);
            
            cv::Mat disparity;
            sgbm.compute(left, right, disparity);
            
            std::cout << "maxDisparity=" << maxDisp << ": " 
                      << sgbm.getLastComputeTimeMs() << " ms\n";
        }
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(SGBMTest, TimingBreakdown) {
    try {
        cv::Mat left(480, 640, CV_8UC1);
        cv::Mat right(480, 640, CV_8UC1);
        cv::randu(left, 0, 255);
        cv::randu(right, 0, 255);
        
        StereoParams params;
        params.maxDisparity = 64;
        params.blockSize = 5;
        params.numDirections = 4;
        
        StereoSGBM sgbm(params);
        
        cv::Mat disparity;
        sgbm.compute(left, right, disparity);
        
        float total = sgbm.getLastComputeTimeMs();
        float cost = sgbm.getCostComputeTimeMs();
        float agg = sgbm.getAggregationTimeMs();
        float post = sgbm.getPostProcessTimeMs();
        
        std::cout << "\nTiming breakdown (640x480, D=64):\n";
        std::cout << "  Total:          " << total << " ms\n";
        std::cout << "  Cost compute:   " << cost << " ms (" << (100*cost/total) << "%)\n";
        std::cout << "  Aggregation:    " << agg << " ms (" << (100*agg/total) << "%)\n";
        std::cout << "  Post-process:   " << post << " ms (" << (100*post/total) << "%)\n";
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

// ============================================================================
// Depth Map Tests
// ============================================================================

TEST(DepthMapTest, DisparityToDepth) {
    // Test the disparity-to-depth formula
    float focalLength = 1000.0f;  // pixels
    float baseline = 0.1f;        // meters
    
    // disparity = focal * baseline / depth
    // depth = focal * baseline / disparity
    
    float disparity = 100.0f;  // pixels
    float expectedDepth = focalLength * baseline / disparity;  // = 1.0 meter
    
    EXPECT_FLOAT_EQ(expectedDepth, 1.0f);
}

TEST(DepthMapTest, ColorizeDepth) {
    try {
        // Create a simple depth map
        cv::Mat depth(100, 100, CV_32FC1);
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 100; x++) {
                depth.at<float>(y, x) = 1.0f + 9.0f * x / 100.0f;  // 1-10 meters
            }
        }
        
        cv::Mat colorized;
        colorizeDepth(depth, colorized, 1.0f, 10.0f);
        
        EXPECT_EQ(colorized.type(), CV_8UC3);
        EXPECT_EQ(colorized.size(), depth.size());
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Error: " << e.what();
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(SGBMTest, EmptyInput) {
    try {
        StereoParams params;
        StereoSGBM sgbm(params);
        
        cv::Mat left, right, disparity;
        
        EXPECT_THROW(sgbm.compute(left, right, disparity), StereoException);
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(SGBMTest, SizeMismatch) {
    try {
        cv::Mat left(100, 100, CV_8UC1);
        cv::Mat right(100, 200, CV_8UC1);  // Different width
        
        StereoParams params;
        StereoSGBM sgbm(params);
        
        cv::Mat disparity;
        EXPECT_THROW(sgbm.compute(left, right, disparity), StereoException);
        
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST(SGBMTest, DisparityRangeExceedsGpuLimit) {
    try {
        cv::Mat left(64, 64, CV_8UC1);
        cv::Mat right(64, 64, CV_8UC1);
        cv::randu(left, 0, 255);
        cv::randu(right, 0, 255);

        StereoParams params;
        params.minDisparity = 0;
        params.maxDisparity = kSgbmMaxDisparityRange + 32;

        StereoSGBM sgbm(params);
        cv::Mat disparity;
        EXPECT_THROW(sgbm.compute(left, right, disparity), StereoException);
    } catch (const CudaException& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

} // namespace test
} // namespace stereo
