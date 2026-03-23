/**
 * calibration.cpp - Stereo camera calibration implementation
 * 
 * This module handles:
 * 1. Checkerboard detection in calibration images
 * 2. Stereo camera calibration (intrinsics + extrinsics)
 * 3. Rectification computation
 * 
 * The calibration process:
 * 1. Capture many images of a checkerboard from different angles
 * 2. Detect corners in each image pair
 * 3. Use OpenCV's stereoCalibrate to estimate parameters
 * 4. Compute rectification transforms
 * 
 * Key OpenCV functions used:
 * - findChessboardCorners: Detect checkerboard corners
 * - cornerSubPix: Refine corner positions to subpixel accuracy
 * - stereoCalibrate: Estimate intrinsics and extrinsics
 * - stereoRectify: Compute rectification transforms
 * - initUndistortRectifyMap: Precompute pixel remap tables
 */

#include "stereo/calibration.h"
#include "stereo/stereo_params.h"
#include "stereo/common.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

namespace stereo {

// ============================================================================
// StereoCalibrator Implementation
// ============================================================================

StereoCalibrator::StereoCalibrator(cv::Size boardSize, float squareSize)
    : boardSize_(boardSize)
    , squareSize_(squareSize)
    , imageSize_(0, 0)
{
}

bool StereoCalibrator::addImagePair(const cv::Mat& left, const cv::Mat& right) {
    // Convert to grayscale if needed
    cv::Mat leftGray, rightGray;
    if (left.channels() == 3) {
        cv::cvtColor(left, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right, rightGray, cv::COLOR_BGR2GRAY);
    } else {
        leftGray = left;
        rightGray = right;
    }
    
    // Store image size (must be consistent across all images)
    if (imageSize_.width == 0) {
        imageSize_ = leftGray.size();
    } else if (imageSize_ != leftGray.size()) {
        std::cerr << "Warning: Image size mismatch! Expected " 
                  << imageSize_ << " but got " << leftGray.size() << "\n";
        return false;
    }
    
    // Find checkerboard corners
    std::vector<cv::Point2f> cornersLeft, cornersRight;
    
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    
    bool foundLeft = cv::findChessboardCorners(leftGray, boardSize_, cornersLeft, flags);
    bool foundRight = cv::findChessboardCorners(rightGray, boardSize_, cornersRight, flags);
    
    if (!foundLeft || !foundRight) {
        std::cout << "Checkerboard not found in " 
                  << (foundLeft ? "right" : "left") << " image\n";
        return false;
    }
    
    // Refine corner positions to subpixel accuracy
    // This is critical for good calibration!
    cv::TermCriteria criteria(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
        30,     // max iterations
        0.001   // epsilon (precision)
    );
    
    cv::cornerSubPix(leftGray, cornersLeft, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    cv::cornerSubPix(rightGray, cornersRight, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    
    // Generate 3D object points (checkerboard in world coordinates)
    // We assume Z=0 (checkerboard is flat) and spacing is squareSize
    std::vector<cv::Point3f> objPts;
    for (int row = 0; row < boardSize_.height; row++) {
        for (int col = 0; col < boardSize_.width; col++) {
            objPts.push_back(cv::Point3f(
                col * squareSize_,
                row * squareSize_,
                0.0f
            ));
        }
    }
    
    // Store the detected corners and object points
    leftCorners_.push_back(cornersLeft);
    rightCorners_.push_back(cornersRight);
    objectPoints_.push_back(objPts);
    
    std::cout << "Added image pair " << numPairs() << " with " 
              << cornersLeft.size() << " corners\n";
    
    return true;
}

CameraParams StereoCalibrator::calibrate() {
    if (numPairs() < 10) {
        std::cerr << "Warning: Only " << numPairs() << " image pairs. "
                  << "Recommend at least 15-20 for good calibration.\n";
    }
    
    if (numPairs() == 0) {
        throw StereoException("No calibration images added!");
    }
    
    CameraParams params;
    params.width = imageSize_.width;
    params.height = imageSize_.height;
    
    // Initialize camera matrices to identity (will be estimated)
    params.K0 = cv::Mat::eye(3, 3, CV_64F);
    params.K1 = cv::Mat::eye(3, 3, CV_64F);
    params.D0 = cv::Mat::zeros(5, 1, CV_64F);
    params.D1 = cv::Mat::zeros(5, 1, CV_64F);
    
    std::cout << "\nRunning stereo calibration on " << numPairs() << " image pairs...\n";
    std::cout << "Image size: " << imageSize_ << "\n";
    std::cout << "Board size: " << boardSize_ << "\n";
    std::cout << "Square size: " << squareSize_ << " mm\n\n";
    
    // Calibration flags
    // CALIB_FIX_INTRINSIC: Use pre-calibrated intrinsics (we don't have them)
    // CALIB_USE_INTRINSIC_GUESS: Start from current K values
    // CALIB_FIX_ASPECT_RATIO: Assume fx == fy
    // CALIB_ZERO_TANGENT_DIST: Assume no tangential distortion
    // CALIB_SAME_FOCAL_LENGTH: Assume both cameras have same focal length
    // CALIB_RATIONAL_MODEL: Use rational distortion model (more coefficients)
    // CALIB_FIX_K3, K4, K5, K6: Fix higher-order distortion coefficients
    
    int flags = cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6;
    
    // Run stereo calibration
    // This estimates:
    // - K0, K1: Intrinsic matrices for both cameras
    // - D0, D1: Distortion coefficients for both cameras
    // - R: Rotation from camera 0 to camera 1
    // - T: Translation from camera 0 to camera 1
    // - E: Essential matrix
    // - F: Fundamental matrix
    
    cv::Mat E, F;  // Essential and Fundamental matrices (we don't need these directly)
    
    double rmsError = cv::stereoCalibrate(
        objectPoints_,
        leftCorners_,
        rightCorners_,
        params.K0, params.D0,
        params.K1, params.D1,
        imageSize_,
        params.R, params.T,
        E, F,
        flags,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6)
    );
    
    std::cout << "Calibration complete!\n";
    std::cout << "RMS reprojection error: " << rmsError << " pixels\n";
    
    if (rmsError > 1.0) {
        std::cerr << "Warning: RMS error > 1.0 indicates poor calibration.\n";
        std::cerr << "Consider recapturing calibration images.\n";
    } else if (rmsError < 0.1) {
        std::cerr << "Warning: RMS error suspiciously low. Check for overfitting.\n";
    }
    
    // Compute derived values
    params.computeDerivedValues();
    
    std::cout << "\nCalibration results:\n";
    std::cout << "  Focal length (left):  fx=" << params.K0.at<double>(0,0) 
              << ", fy=" << params.K0.at<double>(1,1) << " px\n";
    std::cout << "  Principal point (left): cx=" << params.K0.at<double>(0,2)
              << ", cy=" << params.K0.at<double>(1,2) << " px\n";
    std::cout << "  Baseline: " << params.baseline * 1000.0 << " mm\n";
    
    // Compute rectification
    computeRectification(params);
    
    return params;
}

// ============================================================================
// Rectification
// ============================================================================

void computeRectification(CameraParams& params) {
    if (!params.isValid()) {
        throw StereoException("Cannot compute rectification: invalid camera parameters");
    }
    
    cv::Size imageSize(params.width, params.height);
    
    // stereoRectify computes:
    // - R0, R1: Rotation matrices to rectify each camera
    // - P0, P1: Projection matrices in the new coordinate system
    // - Q: Disparity-to-depth mapping matrix
    //
    // After rectification:
    // - Epipolar lines are horizontal (same y-coordinate in both images)
    // - Principal points are aligned vertically
    // - This enables efficient 1D matching
    
    // Alpha parameter controls how much of the original image is preserved
    // alpha = 0: Only valid pixels (some image cropped)
    // alpha = 1: All original pixels (some black borders)
    // alpha = -1: Let OpenCV choose optimal
    double alpha = 0;  // Crop to valid region
    
    cv::Rect validROI0, validROI1;
    
    cv::stereoRectify(
        params.K0, params.D0,
        params.K1, params.D1,
        imageSize,
        params.R, params.T,
        params.R0, params.R1,
        params.P0, params.P1,
        params.Q,
        cv::CALIB_ZERO_DISPARITY,  // Align principal points
        alpha,
        imageSize,  // New image size (same as original)
        &validROI0,
        &validROI1
    );
    
    std::cout << "\nRectification computed.\n";
    std::cout << "Valid ROI (left):  " << validROI0 << "\n";
    std::cout << "Valid ROI (right): " << validROI1 << "\n";
    
    // Precompute remap tables for fast rectification
    // These tables store the source pixel coordinates for each destination pixel
    // This allows rectification with a single lookup per pixel
    
    cv::initUndistortRectifyMap(
        params.K0, params.D0,
        params.R0, params.P0,
        imageSize,
        CV_32FC1,           // Map type (float coordinates)
        params.map0x, params.map0y
    );
    
    cv::initUndistortRectifyMap(
        params.K1, params.D1,
        params.R1, params.P1,
        imageSize,
        CV_32FC1,
        params.map1x, params.map1y
    );
    
    std::cout << "Remap tables computed.\n";
    
    // Extract useful values from Q matrix for depth computation
    // Q maps disparity to 3D point: [X Y Z W]^T = Q * [x y d 1]^T
    // Depth = W / Z = baseline * fx / disparity
    //
    // Q structure:
    // [1  0  0      -cx    ]
    // [0  1  0      -cy    ]
    // [0  0  0       fx    ]
    // [0  0  -1/Tx  (cx-cx')/Tx]
    //
    // where Tx is the baseline (T[0])
    
    double fx = params.Q.at<double>(2, 3);
    double baseline = -1.0 / params.Q.at<double>(3, 2);
    
    std::cout << "Q matrix analysis:\n";
    std::cout << "  Focal length from Q: " << fx << " px\n";
    std::cout << "  Baseline from Q: " << baseline << " m\n";
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Visualize detected corners on an image
 */
cv::Mat drawCorners(const cv::Mat& image, const std::vector<cv::Point2f>& corners, 
                    cv::Size boardSize, bool found) {
    cv::Mat vis;
    if (image.channels() == 1) {
        cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    } else {
        vis = image.clone();
    }
    
    cv::drawChessboardCorners(vis, boardSize, corners, found);
    return vis;
}

/**
 * Apply rectification to a stereo pair
 */
void rectifyImages(const CameraParams& params,
                   const cv::Mat& left, const cv::Mat& right,
                   cv::Mat& leftRect, cv::Mat& rightRect) {
    if (params.map0x.empty() || params.map1x.empty()) {
        throw StereoException("Rectification maps not computed!");
    }
    
    cv::remap(left, leftRect, params.map0x, params.map0y, cv::INTER_LINEAR);
    cv::remap(right, rightRect, params.map1x, params.map1y, cv::INTER_LINEAR);
}

/**
 * Visualize rectification quality by drawing horizontal lines
 * Points on the same scanline should correspond to the same 3D point
 */
cv::Mat visualizeRectification(const cv::Mat& leftRect, const cv::Mat& rightRect) {
    // Create side-by-side image
    cv::Mat combined;
    cv::hconcat(leftRect, rightRect, combined);
    
    // Convert to color if grayscale
    if (combined.channels() == 1) {
        cv::cvtColor(combined, combined, cv::COLOR_GRAY2BGR);
    }
    
    // Draw horizontal lines every 32 pixels
    for (int y = 0; y < combined.rows; y += 32) {
        cv::line(combined, cv::Point(0, y), cv::Point(combined.cols, y),
                 cv::Scalar(0, 255, 0), 1);
    }
    
    return combined;
}

/**
 * Compute reprojection error for a single image pair
 */
double computeReprojectionError(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const cv::Mat& K, const cv::Mat& D,
    const cv::Mat& rvec, const cv::Mat& tvec
) {
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objectPoints, rvec, tvec, K, D, projectedPoints);
    
    double totalError = 0;
    for (size_t i = 0; i < imagePoints.size(); i++) {
        double dx = imagePoints[i].x - projectedPoints[i].x;
        double dy = imagePoints[i].y - projectedPoints[i].y;
        totalError += std::sqrt(dx*dx + dy*dy);
    }
    
    return totalError / imagePoints.size();
}

} // namespace stereo
