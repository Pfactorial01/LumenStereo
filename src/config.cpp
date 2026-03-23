/**
 * config.cpp - Parameter loading and saving
 * 
 * Implements YAML serialization for StereoParams and CameraParams.
 * We use yaml-cpp for parsing because it's clean and well-supported.
 */

#include "stereo/stereo_params.h"
#include <yaml-cpp/yaml.h>
#include <opencv2/core/persistence.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

namespace stereo {

bool loadStereoParams(const std::string& path, StereoParams& params) {
    try {
        YAML::Node config = YAML::LoadFile(path);
        
        if (config["minDisparity"]) params.minDisparity = config["minDisparity"].as<int>();
        if (config["maxDisparity"]) params.maxDisparity = config["maxDisparity"].as<int>();
        if (config["blockSize"]) params.blockSize = config["blockSize"].as<int>();
        if (config["P1"]) params.P1 = config["P1"].as<int>();
        if (config["P2"]) params.P2 = config["P2"].as<int>();
        if (config["uniquenessRatio"]) params.uniquenessRatio = config["uniquenessRatio"].as<int>();
        if (config["disp12MaxDiff"]) params.disp12MaxDiff = config["disp12MaxDiff"].as<int>();
        if (config["preFilterCap"]) params.preFilterCap = config["preFilterCap"].as<int>();
        if (config["useGPU"]) params.useGPU = config["useGPU"].as<bool>();
        if (config["numDirections"]) params.numDirections = config["numDirections"].as<int>();
        if (config["matchingCostMode"]) {
            std::string s = config["matchingCostMode"].as<std::string>();
            if (s == "Census" || s == "census") {
                params.matchingCostMode = MatchingCostMode::Census;
            } else {
                params.matchingCostMode = MatchingCostMode::SAD;
            }
        }
        
        return params.isValid();
    } catch (const std::exception& e) {
        std::cerr << "Failed to load stereo params: " << e.what() << std::endl;
        return false;
    }
}

bool saveStereoParams(const std::string& path, const StereoParams& params) {
    try {
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "minDisparity" << YAML::Value << params.minDisparity;
        out << YAML::Key << "maxDisparity" << YAML::Value << params.maxDisparity;
        out << YAML::Key << "blockSize" << YAML::Value << params.blockSize;
        out << YAML::Key << "P1" << YAML::Value << params.P1;
        out << YAML::Key << "P2" << YAML::Value << params.P2;
        out << YAML::Key << "preFilterCap" << YAML::Value << params.preFilterCap;
        out << YAML::Key << "uniquenessRatio" << YAML::Value << params.uniquenessRatio;
        out << YAML::Key << "disp12MaxDiff" << YAML::Value << params.disp12MaxDiff;
        out << YAML::Key << "useGPU" << YAML::Value << params.useGPU;
        out << YAML::Key << "numDirections" << YAML::Value << params.numDirections;
        out << YAML::Key << "matchingCostMode" << YAML::Value
            << (params.matchingCostMode == MatchingCostMode::Census ? "Census" : "SAD");
        out << YAML::EndMap;
        
        std::ofstream fout(path);
        fout << out.c_str();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save stereo params: " << e.what() << std::endl;
        return false;
    }
}

bool loadCameraParams(const std::string& path, CameraParams& params) {
    try {
        // Use OpenCV's FileStorage for camera matrices (handles Mat serialization)
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open camera params file: " << path << std::endl;
            return false;
        }
        
        fs["width"] >> params.width;
        fs["height"] >> params.height;
        fs["K0"] >> params.K0;
        fs["D0"] >> params.D0;
        fs["K1"] >> params.K1;
        fs["D1"] >> params.D1;
        fs["R"] >> params.R;
        fs["T"] >> params.T;
        
        // Load rectification data if available
        if (!fs["R0"].empty()) fs["R0"] >> params.R0;
        if (!fs["R1"].empty()) fs["R1"] >> params.R1;
        if (!fs["P0"].empty()) fs["P0"] >> params.P0;
        if (!fs["P1"].empty()) fs["P1"] >> params.P1;
        if (!fs["Q"].empty()) fs["Q"] >> params.Q;
        
        params.computeDerivedValues();
        return params.isValid();
    } catch (const std::exception& e) {
        std::cerr << "Failed to load camera params: " << e.what() << std::endl;
        return false;
    }
}

bool saveCameraParams(const std::string& path, const CameraParams& params) {
    try {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            return false;
        }
        
        fs << "width" << params.width;
        fs << "height" << params.height;
        fs << "K0" << params.K0;
        fs << "D0" << params.D0;
        fs << "K1" << params.K1;
        fs << "D1" << params.D1;
        fs << "R" << params.R;
        fs << "T" << params.T;
        fs << "baseline" << params.baseline;
        
        if (!params.R0.empty()) fs << "R0" << params.R0;
        if (!params.R1.empty()) fs << "R1" << params.R1;
        if (!params.P0.empty()) fs << "P0" << params.P0;
        if (!params.P1.empty()) fs << "P1" << params.P1;
        if (!params.Q.empty()) fs << "Q" << params.Q;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save camera params: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Parse Middlebury calib.txt format
 * 
 * Format example:
 *   cam0=[4161.221 0 1445.577; 0 4161.221 984.686; 0 0 1]
 *   cam1=[4161.221 0 1654.636; 0 4161.221 984.686; 0 0 1]
 *   doffs=209.059
 *   baseline=176.252
 *   width=2880
 *   height=1988
 *   ndisp=280
 */
bool loadMiddleburyCalib(const std::string& path, CameraParams& params) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open Middlebury calib file: " << path << std::endl;
        return false;
    }
    
    std::string line;
    double doffs = 0;
    double baseline_mm = 0;
    int ndisp = 0;
    
    // Initialize 3x3 identity matrices
    params.K0 = cv::Mat::eye(3, 3, CV_64F);
    params.K1 = cv::Mat::eye(3, 3, CV_64F);
    params.D0 = cv::Mat::zeros(5, 1, CV_64F);  // No distortion (pre-rectified)
    params.D1 = cv::Mat::zeros(5, 1, CV_64F);
    params.R = cv::Mat::eye(3, 3, CV_64F);      // Identity (pre-rectified)
    params.T = cv::Mat::zeros(3, 1, CV_64F);
    
    while (std::getline(file, line)) {
        // Parse cam0=[...] format
        if (line.find("cam0=") == 0) {
            // Extract matrix values: [fx 0 cx; 0 fy cy; 0 0 1]
            size_t start = line.find('[') + 1;
            size_t end = line.find(']');
            std::string matStr = line.substr(start, end - start);
            
            // Replace semicolons with spaces
            for (char& c : matStr) if (c == ';') c = ' ';
            
            std::istringstream iss(matStr);
            double vals[9];
            for (int i = 0; i < 9; i++) iss >> vals[i];
            
            params.K0.at<double>(0,0) = vals[0];  // fx
            params.K0.at<double>(0,2) = vals[2];  // cx
            params.K0.at<double>(1,1) = vals[4];  // fy
            params.K0.at<double>(1,2) = vals[5];  // cy
        }
        else if (line.find("cam1=") == 0) {
            size_t start = line.find('[') + 1;
            size_t end = line.find(']');
            std::string matStr = line.substr(start, end - start);
            
            for (char& c : matStr) if (c == ';') c = ' ';
            
            std::istringstream iss(matStr);
            double vals[9];
            for (int i = 0; i < 9; i++) iss >> vals[i];
            
            params.K1.at<double>(0,0) = vals[0];
            params.K1.at<double>(0,2) = vals[2];
            params.K1.at<double>(1,1) = vals[4];
            params.K1.at<double>(1,2) = vals[5];
        }
        else if (line.find("doffs=") == 0) {
            doffs = std::stod(line.substr(6));
        }
        else if (line.find("baseline=") == 0) {
            baseline_mm = std::stod(line.substr(9));
        }
        else if (line.find("width=") == 0) {
            params.width = std::stoi(line.substr(6));
        }
        else if (line.find("height=") == 0) {
            params.height = std::stoi(line.substr(7));
        }
        else if (line.find("ndisp=") == 0) {
            ndisp = std::stoi(line.substr(6));
        }
    }
    
    // Middlebury baseline is in mm, convert to meters
    params.baseline = baseline_mm / 1000.0;
    
    // Store disparity offset and count (required for correct search range!)
    params.disparityOffset = static_cast<int>(doffs + 0.5);
    params.numDisparities = ndisp > 0 ? ndisp : 280;
    
    // Set translation vector (baseline along x-axis for rectified stereo)
    params.T.at<double>(0) = -baseline_mm / 1000.0;  // Negative: right cam is to the right
    
    // Compute focal length
    params.focalLength = params.K0.at<double>(0,0);
    
    std::cout << "Loaded Middlebury calibration:" << std::endl;
    std::cout << "  Resolution: " << params.width << "x" << params.height << std::endl;
    std::cout << "  Focal length: " << params.focalLength << " px" << std::endl;
    std::cout << "  Baseline: " << params.baseline * 1000 << " mm" << std::endl;
    std::cout << "  Disparity offset: " << doffs << " px" << std::endl;
    std::cout << "  Num disparities: " << ndisp << std::endl;
    
    return params.width > 0 && params.height > 0;
}

} // namespace stereo
