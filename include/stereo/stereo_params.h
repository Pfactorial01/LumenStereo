#pragma once

/**
 * stereo_params.h - Configuration structures for the stereo pipeline
 * 
 * This file defines:
 * 1. StereoParams - SGBM algorithm parameters
 * 2. CameraParams - Calibration and camera intrinsics
 * 3. Functions to load/save from YAML
 */

#include <string>
#include <opencv2/core.hpp>

namespace stereo {

/**
 * Pixel matching cost before SGM aggregation
 */
enum class MatchingCostMode : int {
    SAD = 0,     // Mean absolute difference over blockSize x blockSize window
    Census = 1,  // 5x5 sparse census + Hamming distance (robust to exposure)
};

/**
 * SGBM Algorithm Parameters
 * 
 * These control the disparity computation quality/speed tradeoff.
 * 
 * Key parameters explained:
 * 
 * minDisparity/maxDisparity: 
 *   The search range for matching. If you know objects are between 0.5m and 10m,
 *   you can limit this range for faster computation.
 *   Disparity is inversely proportional to depth: close objects have HIGH disparity.
 * 
 * blockSize:
 *   Size of the matching window (must be odd: 3, 5, 7, ...).
 *   Larger = more robust but loses detail at edges.
 *   Smaller = more detail but more noise.
 *   Typical: 5-11 for indoor, 3-7 for textured scenes.
 * 
 * P1, P2:
 *   Smoothness penalties for the semi-global matching.
 *   P1 = penalty for disparity changes of 1 (small penalty, allows edges)
 *   P2 = penalty for disparity changes > 1 (large penalty, enforces smoothness)
 *   Rule: P2 > P1. Typical: P1 = 8*channels*blockSize^2, P2 = 32*channels*blockSize^2
 * 
 * uniquenessRatio:
 *   If the best match isn't significantly better than second-best, mark as invalid.
 *   Higher = stricter filtering. Typical: 5-15.
 * 
 * disp12MaxDiff:
 *   Maximum allowed difference in left-right consistency check.
 *   Set to 1 for strict checking, -1 to disable.
 */
struct StereoParams {
    // Disparity search range
    int minDisparity = 0;       // Minimum disparity (usually 0)
    int maxDisparity = 128;     // Maximum disparity (balance memory/coverage)
    
    // Block matching parameters  
    int blockSize = 5;          // Window size for matching (odd number)
    
    // Smoothness penalties (Semi-Global Matching)
    int P1 = 200;               // Penalty for ±1 disparity change (default: 8*blockSize²)
    int P2 = 800;               // Penalty for larger disparity changes (default: 32*blockSize²)
    
    // Pre-processing
    int preFilterCap = 31;      // Sobel prefilter clip value (bounds per-pixel SAD cost)
    
    // Post-processing
    int uniquenessRatio = 10;   // Uniqueness threshold (percentage)
    int disp12MaxDiff = 1;      // Max L-R consistency difference (-1 to disable)
    int speckleWindowSize = 100; // Speckle filter window
    int speckleRange = 32;       // Max disparity variation in speckle region
    
    // Algorithm selection
    bool useGPU = true;          // Use CUDA implementation
    bool useSharedMemory = true; // Use shared memory for cost computation (SAD only)
    int numDirections = 4;       // Number of SGM paths (4 or 8)
    MatchingCostMode matchingCostMode = MatchingCostMode::SAD;
    
    /**
     * Compute reasonable P1/P2 values based on block size
     * This formula is from the original SGM paper
     */
    void autoComputePenalties(int numChannels = 1) {
        P1 = 8 * numChannels * blockSize * blockSize;
        P2 = 32 * numChannels * blockSize * blockSize;
    }
    
    /**
     * Validate parameters
     */
    bool isValid() const {
        if (blockSize < 1 || blockSize % 2 == 0) return false;  // Must be odd
        if (minDisparity < 0) return false;
        if (maxDisparity <= minDisparity) return false;
        if (P2 <= P1) return false;
        if (uniquenessRatio < 0 || uniquenessRatio > 100) return false;
        return true;
    }
};

/**
 * Camera Calibration Parameters
 * 
 * These come from the calibration process and describe:
 * 1. How the lens distorts the image (distortion coefficients)
 * 2. The camera's "zoom" and center point (intrinsic matrix K)
 * 3. How the two cameras are positioned relative to each other (R, T)
 * 
 * Intrinsic Matrix K (3x3):
 *   [fx  0  cx]
 *   [0  fy  cy]
 *   [0   0   1]
 * 
 *   fx, fy = focal length in pixels (how "zoomed in" the camera is)
 *   cx, cy = principal point (where the optical axis hits the sensor)
 * 
 * Distortion Coefficients D:
 *   [k1, k2, p1, p2, k3, ...]
 *   k1, k2, k3 = radial distortion (barrel/pincushion effect)
 *   p1, p2 = tangential distortion (sensor not parallel to lens)
 * 
 * Extrinsics (R, T):
 *   R = 3x3 rotation matrix from camera 0 to camera 1
 *   T = 3x1 translation vector from camera 0 to camera 1
 *   baseline = ||T|| = distance between cameras (critical for depth!)
 */
struct CameraParams {
    // Image size
    int width = 0;
    int height = 0;
    
    // Camera 0 (left) intrinsics
    cv::Mat K0;         // 3x3 intrinsic matrix
    cv::Mat D0;         // Distortion coefficients
    
    // Camera 1 (right) intrinsics  
    cv::Mat K1;         // 3x3 intrinsic matrix
    cv::Mat D1;         // Distortion coefficients
    
    // Extrinsics (relationship between cameras)
    cv::Mat R;          // 3x3 rotation matrix
    cv::Mat T;          // 3x1 translation vector
    
    // Derived values
    double baseline = 0.0;      // Distance between cameras in meters
    double focalLength = 0.0;   // Focal length in pixels (average of fx, fy)
    
    // Middlebury-style: disparity search range (doffs = min disparity, ndisp = range)
    int disparityOffset = 0;    // Minimum disparity in scene (e.g. 209 for Adirondack)
    int numDisparities = 0;     // Number of disparities (e.g. 280)
    
    // Rectification outputs (computed from above)
    cv::Mat R0, R1;     // Rectification rotation for each camera
    cv::Mat P0, P1;     // Projection matrices after rectification
    cv::Mat Q;          // Disparity-to-depth mapping matrix
    
    // Precomputed rectification maps (for fast remapping)
    cv::Mat map0x, map0y;   // Remap coordinates for left camera
    cv::Mat map1x, map1y;   // Remap coordinates for right camera
    
    /**
     * Compute baseline from translation vector
     */
    void computeDerivedValues() {
        if (!T.empty()) {
            baseline = cv::norm(T);
        }
        if (!K0.empty()) {
            focalLength = (K0.at<double>(0,0) + K0.at<double>(1,1)) / 2.0;
        }
    }
    
    /**
     * Check if calibration is loaded
     */
    bool isValid() const {
        return !K0.empty() && !K1.empty() && !R.empty() && !T.empty() 
               && width > 0 && height > 0;
    }
};

/**
 * Load stereo parameters from YAML file
 */
bool loadStereoParams(const std::string& path, StereoParams& params);

/**
 * Save stereo parameters to YAML file
 */
bool saveStereoParams(const std::string& path, const StereoParams& params);

/**
 * Load camera calibration from YAML file
 */
bool loadCameraParams(const std::string& path, CameraParams& params);

/**
 * Save camera calibration to YAML file
 */
bool saveCameraParams(const std::string& path, const CameraParams& params);

/**
 * Load Middlebury calib.txt format
 * This is used by the Adirondack-perfect dataset
 */
bool loadMiddleburyCalib(const std::string& path, CameraParams& params);

} // namespace stereo
