#pragma once

/**
 * calibration.h - Stereo camera calibration interface
 * 
 * This module provides:
 * 1. StereoCalibrator - Class for collecting images and running calibration
 * 2. Rectification computation
 * 3. Utility functions for visualization
 * 
 * Calibration workflow:
 * 1. Print a checkerboard pattern (e.g., 9x6 inner corners)
 * 2. Capture 15-30 image pairs with the board at different positions/angles
 * 3. Run calibration to estimate camera parameters
 * 4. Save calibration to YAML for later use
 * 
 * Tips for good calibration:
 * - Cover the entire field of view with checkerboard positions
 * - Include tilted views (not just fronto-parallel)
 * - Keep the board in focus
 * - Use a rigid, flat checkerboard (not paper that can bend)
 * - Ensure good lighting without reflections
 */

#include "stereo_params.h"
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace stereo {

/**
 * Stereo camera calibrator
 * 
 * Usage:
 *   StereoCalibrator calibrator(cv::Size(9, 6), 25.0f);  // 9x6 board, 25mm squares
 *   calibrator.addImagePair(left1, right1);
 *   calibrator.addImagePair(left2, right2);
 *   // ... add more pairs (15-30 recommended) ...
 *   CameraParams params = calibrator.calibrate();
 *   saveCameraParams("calibration.yaml", params);
 */
class StereoCalibrator {
public:
    /**
     * @param boardSize Number of inner corners (cols x rows)
     *                  For a 10x7 checkerboard, use cv::Size(9, 6)
     * @param squareSize Size of each square in millimeters
     */
    StereoCalibrator(cv::Size boardSize, float squareSize);
    
    /**
     * Add a stereo image pair for calibration
     * 
     * @param left  Left camera image
     * @param right Right camera image  
     * @return true if checkerboard found in both images
     * 
     * The function:
     * 1. Converts to grayscale if needed
     * 2. Detects checkerboard corners
     * 3. Refines corners to subpixel accuracy
     * 4. Stores corners for later calibration
     */
    bool addImagePair(const cv::Mat& left, const cv::Mat& right);
    
    /**
     * Run stereo calibration on collected image pairs
     * 
     * @return Calibration parameters (intrinsics, extrinsics, rectification)
     * @throws StereoException if no images added
     * 
     * The function:
     * 1. Runs OpenCV's stereoCalibrate
     * 2. Computes rectification transforms
     * 3. Precomputes remap tables
     */
    CameraParams calibrate();
    
    /**
     * Get number of valid image pairs added
     */
    int numPairs() const { return static_cast<int>(leftCorners_.size()); }
    
    /**
     * Get the image size (from first added pair)
     */
    cv::Size getImageSize() const { return imageSize_; }
    
    /**
     * Get board size
     */
    cv::Size getBoardSize() const { return boardSize_; }
    
private:
    cv::Size boardSize_;
    float squareSize_;
    cv::Size imageSize_;
    
    std::vector<std::vector<cv::Point2f>> leftCorners_;
    std::vector<std::vector<cv::Point2f>> rightCorners_;
    std::vector<std::vector<cv::Point3f>> objectPoints_;
};

/**
 * Compute rectification transforms from calibration
 * 
 * This function:
 * 1. Calls cv::stereoRectify to compute R0, R1, P0, P1, Q
 * 2. Calls cv::initUndistortRectifyMap to precompute remap tables
 * 
 * After calling this, you can use rectifyImages() for fast rectification.
 */
void computeRectification(CameraParams& params);

/**
 * Apply rectification to a stereo pair using precomputed maps
 */
void rectifyImages(const CameraParams& params,
                   const cv::Mat& left, const cv::Mat& right,
                   cv::Mat& leftRect, cv::Mat& rightRect);

/**
 * Visualize detected corners on an image
 * 
 * @param image      Input image
 * @param corners    Detected corner positions
 * @param boardSize  Checkerboard size (cols x rows)
 * @param found      Whether detection was successful
 * @return Color image with corners drawn
 */
cv::Mat drawCorners(const cv::Mat& image, const std::vector<cv::Point2f>& corners,
                    cv::Size boardSize, bool found);

/**
 * Visualize rectification quality
 * 
 * Creates a side-by-side image with horizontal lines.
 * After good rectification, corresponding points should be on the same scanline.
 * 
 * @param leftRect   Rectified left image
 * @param rightRect  Rectified right image
 * @return Combined image with horizontal guide lines
 */
cv::Mat visualizeRectification(const cv::Mat& leftRect, const cv::Mat& rightRect);

/**
 * Compute reprojection error for validation
 */
double computeReprojectionError(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const cv::Mat& K, const cv::Mat& D,
    const cv::Mat& rvec, const cv::Mat& tvec
);

} // namespace stereo
