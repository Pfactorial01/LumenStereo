/**
 * point_cloud.cpp - Point cloud generation and PLY export
 */

#include "stereo/point_cloud.h"
#include <fstream>
#include <iostream>

namespace stereo {

std::vector<Point3D> depthToPointCloud(
    const cv::Mat& depth,
    const cv::Mat& color,
    const CameraParams& params
) {
    std::vector<Point3D> points;
    
    // Get camera intrinsics
    double fx = params.K0.at<double>(0, 0);
    double fy = params.K0.at<double>(1, 1);
    double cx = params.K0.at<double>(0, 2);
    double cy = params.K0.at<double>(1, 2);
    
    bool hasColor = !color.empty() && color.size() == depth.size();
    
    // Reserve approximate space (many pixels may be invalid)
    points.reserve(depth.rows * depth.cols / 4);
    
    for (int v = 0; v < depth.rows; v++) {
        for (int u = 0; u < depth.cols; u++) {
            float Z = depth.at<float>(v, u);
            
            // Skip invalid depths
            if (Z <= 0 || Z > 100.0f) continue;
            
            // Back-project to 3D
            float X = static_cast<float>((u - cx) * Z / fx);
            float Y = static_cast<float>((v - cy) * Z / fy);
            
            Point3D pt(X, Y, Z);
            
            // Add color if available
            if (hasColor) {
                cv::Vec3b bgr = color.at<cv::Vec3b>(v, u);
                pt.r = bgr[2];  // BGR to RGB
                pt.g = bgr[1];
                pt.b = bgr[0];
            }
            
            points.push_back(pt);
        }
    }
    
    return points;
}

bool exportPLY(const std::vector<Point3D>& points,
               const std::string& filename,
               bool binary) {
    if (binary) {
        // Binary PLY
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        
        // Write header
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "element vertex " << points.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "end_header\n";
        
        // Write binary data
        for (const auto& pt : points) {
            file.write(reinterpret_cast<const char*>(&pt.x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pt.y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pt.z), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pt.r), sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(&pt.g), sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(&pt.b), sizeof(uint8_t));
        }
        
        return true;
    } else {
        return exportPLY_ASCII(points, filename);
    }
}

bool exportPLY_ASCII(const std::vector<Point3D>& points,
                     const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Write points
    for (const auto& pt : points) {
        file << pt.x << " " << pt.y << " " << pt.z << " "
             << static_cast<int>(pt.r) << " "
             << static_cast<int>(pt.g) << " "
             << static_cast<int>(pt.b) << "\n";
    }
    
    return true;
}

} // namespace stereo
