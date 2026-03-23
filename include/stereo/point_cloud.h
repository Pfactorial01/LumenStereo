#pragma once

/**
 * point_cloud.h - Point cloud generation and export
 */

#include "stereo_params.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace stereo {

/**
 * 3D point with optional color
 */
struct Point3D {
    float x, y, z;
    uint8_t r, g, b;
    
    Point3D() : x(0), y(0), z(0), r(255), g(255), b(255) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_), r(255), g(255), b(255) {}
    Point3D(float x_, float y_, float z_, uint8_t r_, uint8_t g_, uint8_t b_)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
};

/**
 * Generate point cloud from depth map
 * 
 * For each pixel (u, v) with depth Z:
 *   X = (u - cx) * Z / fx
 *   Y = (v - cy) * Z / fy
 * 
 * @param depth Depth map in meters (CV_32FC1)
 * @param color Optional color image (CV_8UC3)
 * @param params Camera parameters (for cx, cy, fx, fy)
 * @return Vector of 3D points
 */
std::vector<Point3D> depthToPointCloud(
    const cv::Mat& depth,
    const cv::Mat& color,
    const CameraParams& params
);

/**
 * Export point cloud to PLY file
 * 
 * @param points Point cloud
 * @param filename Output filename
 * @param binary If true, write binary PLY (smaller, faster)
 */
bool exportPLY(const std::vector<Point3D>& points,
               const std::string& filename,
               bool binary = true);

/**
 * Export point cloud to PLY (ASCII format)
 */
bool exportPLY_ASCII(const std::vector<Point3D>& points,
                     const std::string& filename);

} // namespace stereo
