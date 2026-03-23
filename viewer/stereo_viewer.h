#pragma once

/**
 * stereo_viewer.h - Interactive ImGui-based stereo depth viewer
 * 
 * Features:
 * - 4-panel display: Left, Right, Disparity, Depth
 * - Parameter sliders for real-time SGBM tuning
 * - Click-to-probe depth values
 * - FPS and timing overlay
 */

#include "stereo/stereo_params.h"
#include <opencv2/core.hpp>
#include <string>
#include <functional>

// Forward declarations (avoid including GLFW/OpenGL in header)
struct GLFWwindow;

namespace stereo {

/**
 * OpenGL texture wrapper for displaying images
 */
class GLTexture {
public:
    GLTexture();
    ~GLTexture();
    
    // Upload image data to GPU texture
    void upload(const cv::Mat& image);
    
    // Get OpenGL texture ID (for ImGui::Image)
    unsigned int id() const { return textureId_; }
    
    int width() const { return width_; }
    int height() const { return height_; }
    
private:
    unsigned int textureId_ = 0;
    int width_ = 0;
    int height_ = 0;
};

/**
 * Interactive stereo viewer with ImGui
 */
class StereoViewer {
public:
    explicit StereoViewer(const std::string& title = "LumenStereo Viewer");
    ~StereoViewer();
    
    // Initialize window and OpenGL context
    bool init(int width = 1600, int height = 900);
    
    // Clean up resources
    void shutdown();
    
    // Update images to display
    void setImages(const cv::Mat& left, const cv::Mat& right,
                   const cv::Mat& disparity, const cv::Mat& depthColor);
    
    // Set raw disparity for click-to-probe (CV_16SC1)
    void setDisparityRaw(const cv::Mat& disparityRaw);
    
    // Set camera parameters for depth calculation
    void setCameraParams(const CameraParams& params);
    
    // Set timing info
    void setTiming(float totalMs, float costMs, float aggMs, float postMs);
    
    // Get current stereo parameters (modified by sliders)
    StereoParams& getParams() { return params_; }
    const StereoParams& getParams() const { return params_; }
    
    // Check if parameters were changed by user
    bool paramsChanged() const { return paramsChanged_; }
    void clearParamsChanged() { paramsChanged_ = false; }
    
    // Render one frame (call in loop)
    // Returns false if window should close
    bool render();
    
    // Check if window is open
    bool isOpen() const;
    
    // Callback for when user requests recompute
    using RecomputeCallback = std::function<void()>;
    void setRecomputeCallback(RecomputeCallback cb) { recomputeCallback_ = cb; }

private:
    void renderMenuBar();
    void renderControlPanel();
    void renderImagePanel(const char* title, GLTexture& texture, int panelWidth, int panelHeight);
    void renderInfoPanel();
    void renderProbeOverlay();
    
    // Handle mouse click for probing
    void handleMouseClick(double xpos, double ypos);
    
    std::string title_;
    GLFWwindow* window_ = nullptr;
    
    // OpenGL textures for each image
    GLTexture texLeft_, texRight_, texDisparity_, texDepth_;
    
    // Stereo parameters (editable via sliders)
    StereoParams params_;
    bool paramsChanged_ = false;
    
    // Camera parameters for depth calculation
    CameraParams camParams_;
    
    // Raw disparity for click-to-probe (CV_16SC1)
    cv::Mat disparityRaw_;
    
    // Probe state
    bool probeActive_ = false;
    int probeX_ = 0, probeY_ = 0;
    float probeDisparity_ = 0;
    float probeDepth_ = 0;
    
    // Timing info
    float totalTimeMs_ = 0;
    float costTimeMs_ = 0;
    float aggTimeMs_ = 0;
    float postTimeMs_ = 0;
    
    // FPS tracking
    float fps_ = 0;
    int frameCount_ = 0;
    double lastFpsTime_ = 0;
    
    // Display options
    bool showLeftRight_ = true;
    bool showDisparity_ = true;
    bool showDepth_ = true;
    int displayScale_ = 50;  // Percentage
    
    // Recompute callback
    RecomputeCallback recomputeCallback_;
};

} // namespace stereo
