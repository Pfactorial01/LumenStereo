/**
 * stereo_viewer.cpp - Interactive ImGui-based stereo depth viewer
 */

#include "stereo_viewer.h"
#include "stereo/sgbm_limits.h"

// OpenGL/GLFW (must be before ImGui)
#include <GL/gl.h>
#include <GLFW/glfw3.h>

// CUDA-OpenGL interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

namespace stereo {

// ============================================================================
// GLTexture Implementation
// ============================================================================

GLTexture::GLTexture() {
    // Don't create texture here - OpenGL context may not exist yet
    textureId_ = 0;
}

GLTexture::~GLTexture() {
    if (textureId_ != 0) {
        glDeleteTextures(1, &textureId_);
    }
}

void GLTexture::upload(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "GLTexture::upload - empty image\n";
        return;
    }
    
    // Create texture on first upload (when OpenGL context exists)
    if (textureId_ == 0) {
        glGenTextures(1, &textureId_);
        glBindTexture(GL_TEXTURE_2D, textureId_);
        
        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    
    width_ = image.cols;
    height_ = image.rows;
    
    // Convert to RGB if needed (OpenGL expects RGB, OpenCV uses BGR)
    cv::Mat rgb;
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    } else {
        rgb = image;
    }
    
    // Ensure 8-bit
    cv::Mat rgb8;
    if (rgb.depth() != CV_8U) {
        rgb.convertTo(rgb8, CV_8U);
    } else {
        rgb8 = rgb;
    }
    
    // Make sure data is continuous
    if (!rgb8.isContinuous()) {
        rgb8 = rgb8.clone();
    }
    
    glBindTexture(GL_TEXTURE_2D, textureId_);
    
    // Set pixel alignment (important for non-power-of-2 textures)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width_, height_, 0, 
                 GL_RGB, GL_UNSIGNED_BYTE, rgb8.data);
    
}

// ============================================================================
// StereoViewer Implementation
// ============================================================================

StereoViewer::StereoViewer(const std::string& title)
    : title_(title)
{
    params_.maxDisparity = 128;
    params_.blockSize = 5;
    params_.P1 = 200;
    params_.P2 = 800;
    params_.preFilterCap = 31;
    params_.uniquenessRatio = 5;
    params_.disp12MaxDiff = 2;
    params_.speckleWindowSize = 0;
    params_.speckleRange = 24;
}

StereoViewer::~StereoViewer() {
    shutdown();
}

bool StereoViewer::init(int width, int height) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }
    
    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    
    // Create window
    window_ = glfwCreateWindow(width, height, title_.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);  // Enable vsync
    
    // Initialize CUDA with the OpenGL device
    // This is important for CUDA-OpenGL interoperability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found\n";
    } else {
        // Reset CUDA to use the primary device
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        } else {
            // Force CUDA runtime initialization
            cudaFree(nullptr);
        }
    }
    
    // Store this pointer for callbacks
    glfwSetWindowUserPointer(window_, this);
    
    // Mouse button callback for click-to-probe
    glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            double xpos, ypos;
            glfwGetCursorPos(w, &xpos, &ypos);
            auto* viewer = static_cast<StereoViewer*>(glfwGetWindowUserPointer(w));
            if (viewer && !ImGui::GetIO().WantCaptureMouse) {
                viewer->handleMouseClick(xpos, ypos);
            }
        }
    });
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    
    lastFpsTime_ = glfwGetTime();
    
    std::cout << "Viewer initialized: " << width << "x" << height << "\n";
    return true;
}

void StereoViewer::shutdown() {
    if (window_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

bool StereoViewer::isOpen() const {
    return window_ && !glfwWindowShouldClose(window_);
}

void StereoViewer::setImages(const cv::Mat& left, const cv::Mat& right,
                             const cv::Mat& disparity, const cv::Mat& depthColor) {
    // Upload to GPU textures (full resolution)
    if (!left.empty()) {
        texLeft_.upload(left);
    }
    if (!right.empty()) {
        texRight_.upload(right);
    }
    
    // Normalize disparity for display: invalid = black, valid = full colormap
    if (!disparity.empty()) {
        cv::Mat dispVis(disparity.size(), CV_8UC1);
        dispVis.setTo(0);

        if (disparity.type() == CV_16SC1) {
            // Manual min/max over valid pixels (OpenCV minMaxLoc mask can be unreliable)
            int16_t minVal = 32767, maxVal = -32768;
            for (int y = 0; y < disparity.rows; y++) {
                const int16_t* row = disparity.ptr<int16_t>(y);
                for (int x = 0; x < disparity.cols; x++) {
                    int16_t d = row[x];
                    if (d > 0) {
                        if (d < minVal) minVal = d;
                        if (d > maxVal) maxVal = d;
                    }
                }
            }

            if (maxVal > minVal) {
                float scale = 255.0f / (maxVal - minVal);
                for (int y = 0; y < disparity.rows; y++) {
                    const int16_t* row = disparity.ptr<int16_t>(y);
                    unsigned char* out = dispVis.ptr<unsigned char>(y);
                    for (int x = 0; x < disparity.cols; x++) {
                        int16_t d = row[x];
                        if (d > 0) {
                            float norm = (static_cast<float>(d) - minVal) * scale;
                            out[x] = static_cast<unsigned char>(std::max(0.f, std::min(255.f, norm)));
                        }
                    }
                }
            } else {
                // Fallback: fixed scale from min/max disparity, invalid stays 0
                int range = (params_.maxDisparity - params_.minDisparity) * 16;
                double scale = (range > 0) ? (255.0 / range) : 0;
                int minRaw = params_.minDisparity * 16;
                for (int y = 0; y < disparity.rows; y++) {
                    const int16_t* row = disparity.ptr<int16_t>(y);
                    unsigned char* out = dispVis.ptr<unsigned char>(y);
                    for (int x = 0; x < disparity.cols; x++) {
                        int16_t d = row[x];
                        if (d > 0) {
                            int v = static_cast<int>((d - minRaw) * scale);
                            out[x] = static_cast<unsigned char>(std::max(0, std::min(255, v)));
                        }
                    }
                }
            }
        } else {
            disparity.convertTo(dispVis, CV_8U);
        }

        cv::Mat dispColor;
        cv::applyColorMap(dispVis, dispColor, cv::COLORMAP_TURBO);
        dispColor.setTo(cv::Scalar(0, 0, 0), dispVis == 0);
        texDisparity_.upload(dispColor);
    }
    
    if (!depthColor.empty()) {
        texDepth_.upload(depthColor);
    }
}

void StereoViewer::setDisparityRaw(const cv::Mat& disparityRaw) {
    disparityRaw_ = disparityRaw.clone();
}

void StereoViewer::setCameraParams(const CameraParams& params) {
    camParams_ = params;
    // Middlebury datasets provide `doffs` (principal point x-offset). Ground-truth disparities
    // are typically reported without this offset, and the depth formula uses (disp + doffs).
    // Keep the matcher range starting at 0 and apply doffs during depth conversion.
    if (params.disparityOffset > 0) {
        params_.minDisparity = 0;
        int range = middleburyMatcherMaxDisparity(params.numDisparities);
        params_.maxDisparity = range;
        params_.P1 = 200;
        params_.P2 = 800;
        params_.preFilterCap = 31;
        params_.uniquenessRatio = 5;
        params_.disp12MaxDiff = 3;
        params_.speckleWindowSize = 0;
        params_.speckleRange = 24;
        params_.numDirections = 8;
        params_.matchingCostMode = MatchingCostMode::Census;
    }
}

void StereoViewer::setTiming(float totalMs, float costMs, float aggMs, float postMs) {
    totalTimeMs_ = totalMs;
    costTimeMs_ = costMs;
    aggTimeMs_ = aggMs;
    postTimeMs_ = postMs;
}

bool StereoViewer::render() {
    if (!isOpen()) return false;
    
    glfwPollEvents();
    
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Calculate FPS
    frameCount_++;
    double currentTime = glfwGetTime();
    if (currentTime - lastFpsTime_ >= 1.0) {
        fps_ = static_cast<float>(frameCount_) / static_cast<float>(currentTime - lastFpsTime_);
        frameCount_ = 0;
        lastFpsTime_ = currentTime;
    }
    
    // Render UI components
    renderMenuBar();
    renderControlPanel();
    renderInfoPanel();
    
    // Get window size for image layout
    int windowWidth, windowHeight;
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);
    
    // Image panels (2x2 grid on the right side)
    int controlPanelWidth = 300;
    int availableWidth = windowWidth - controlPanelWidth - 20;
    int panelWidth = availableWidth / 2;
    int panelHeight = (windowHeight - 60) / 2;  // Leave space for menu bar
    
    // Position image windows
    ImGui::SetNextWindowPos(ImVec2(static_cast<float>(controlPanelWidth + 10), 30));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(panelWidth), static_cast<float>(panelHeight)));
    renderImagePanel("Left Image", texLeft_, panelWidth, panelHeight);
    
    ImGui::SetNextWindowPos(ImVec2(static_cast<float>(controlPanelWidth + 10 + panelWidth), 30));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(panelWidth), static_cast<float>(panelHeight)));
    renderImagePanel("Right Image", texRight_, panelWidth, panelHeight);
    
    ImGui::SetNextWindowPos(ImVec2(static_cast<float>(controlPanelWidth + 10), static_cast<float>(30 + panelHeight)));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(panelWidth), static_cast<float>(panelHeight)));
    renderImagePanel("Disparity", texDisparity_, panelWidth, panelHeight);
    
    ImGui::SetNextWindowPos(ImVec2(static_cast<float>(controlPanelWidth + 10 + panelWidth), static_cast<float>(30 + panelHeight)));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(panelWidth), static_cast<float>(panelHeight)));
    renderImagePanel("Depth (Colorized)", texDepth_, panelWidth, panelHeight);
    
    // Probe overlay
    renderProbeOverlay();
    
    // Render ImGui
    ImGui::Render();
    
    // OpenGL rendering
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    glfwSwapBuffers(window_);
    
    return !glfwWindowShouldClose(window_);
}

void StereoViewer::renderMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            // Disabled items shown grayed out
            ImGui::TextDisabled("Load Stereo Pair... (TODO)");
            ImGui::TextDisabled("Save Disparity... (TODO)");
            ImGui::TextDisabled("Save Point Cloud... (TODO)");
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Esc")) {
                glfwSetWindowShouldClose(window_, true);
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Show Left/Right", nullptr, &showLeftRight_);
            ImGui::MenuItem("Show Disparity", nullptr, &showDisparity_);
            ImGui::MenuItem("Show Depth", nullptr, &showDepth_);
            ImGui::Separator();
            ImGui::SliderInt("Display Scale", &displayScale_, 25, 100, "%d%%");
            ImGui::EndMenu();
        }
        
        // Show image info in menu bar
        ImGui::Separator();
        if (texLeft_.width() > 0) {
            ImGui::Text("Image: %dx%d", texLeft_.width(), texLeft_.height());
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "No images loaded");
        }
        
        ImGui::EndMainMenuBar();
    }
}

void StereoViewer::renderControlPanel() {
    ImGui::SetNextWindowPos(ImVec2(10, 30));
    ImGui::SetNextWindowSize(ImVec2(280, 600));
    
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    
    // SGBM Parameters
    if (ImGui::CollapsingHeader("SGBM Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool changed = false;

        int costModeInt = static_cast<int>(params_.matchingCostMode);
        const char* costNames[] = { "SAD (block)", "Census (5x5)" };
        if (ImGui::Combo("Matching cost", &costModeInt, costNames, 2)) {
            params_.matchingCostMode = static_cast<MatchingCostMode>(costModeInt);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Census is usually better on Middlebury; SAD uses blockSize window.");
        }
        
        // Disparity range
        ImGui::Text("Disparity Range");
        int maxDisp = params_.maxDisparity;
        if (ImGui::SliderInt("Max Disparity", &maxDisp, 16, stereo::kSgbmMaxDisparityRange)) {
            // Round to multiple of 16 for efficiency
            params_.maxDisparity = (maxDisp / 16) * 16;
            if (params_.maxDisparity < 16) params_.maxDisparity = 16;
            changed = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Search range for matching.\nHigher = closer objects visible, but slower.");
        }
        
        // Block size
        int blockSize = params_.blockSize;
        if (ImGui::SliderInt("Block Size", &blockSize, 3, 21)) {
            // Must be odd
            params_.blockSize = (blockSize / 2) * 2 + 1;
            changed = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Size of matching window.\nLarger = more robust, less detail.");
        }
        
        ImGui::Separator();
        ImGui::Text("Smoothness Penalties");
        
        // P1 penalty
        if (ImGui::SliderInt("P1 (small change)", &params_.P1, 1, 500)) {
            changed = true;
        }
        
        // P2 penalty
        if (ImGui::SliderInt("P2 (large change)", &params_.P2, 1, 2000)) {
            changed = true;
        }
        
        // Auto-compute penalties button
        if (ImGui::Button("Auto P1/P2")) {
            params_.autoComputePenalties();
            changed = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Compute P1/P2 from block size using the\nstandard formula from the SGM paper.");
        }
        
        ImGui::Separator();
        ImGui::Text("Post-Processing");
        
        if (ImGui::SliderInt("Uniqueness %%", &params_.uniquenessRatio, 0, 30)) {
            changed = true;
        }
        
        if (changed) {
            paramsChanged_ = true;
        }
    }
    
    ImGui::Separator();
    
    // Recompute button
    if (ImGui::Button("Recompute Disparity", ImVec2(-1, 40))) {
        if (recomputeCallback_) {
            recomputeCallback_();
        }
        paramsChanged_ = false;
    }
    
    if (paramsChanged_) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Parameters changed - click to apply");
    }
    
    ImGui::Separator();
    
    // Camera info
    if (ImGui::CollapsingHeader("Camera Info")) {
        ImGui::Text("Resolution: %dx%d", camParams_.width, camParams_.height);
        ImGui::Text("Focal Length: %.1f px", camParams_.focalLength);
        ImGui::Text("Baseline: %.1f mm", camParams_.baseline * 1000);
    }
    
    ImGui::End();
}

void StereoViewer::renderImagePanel(const char* title, GLTexture& texture, int panelWidth, int panelHeight) {
    ImGui::Begin(title, nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    
    if (texture.id() != 0 && texture.width() > 0) {
        // Calculate image size to fit in panel while maintaining aspect ratio
        float aspect = static_cast<float>(texture.width()) / texture.height();
        ImVec2 available = ImGui::GetContentRegionAvail();
        
        float imgWidth = available.x - 10;  // Leave some margin
        float imgHeight = imgWidth / aspect;
        
        if (imgHeight > available.y - 10) {
            imgHeight = available.y - 10;
            imgWidth = imgHeight * aspect;
        }
        
        // Ensure minimum size
        if (imgWidth < 50) imgWidth = 50;
        if (imgHeight < 50) imgHeight = 50;
        
        // Center the image
        float offsetX = (available.x - imgWidth) / 2;
        if (offsetX > 0) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
        }
        
        // Display image with flipped V coordinates (OpenGL has origin at bottom-left)
        // UV0 = top-left, UV1 = bottom-right
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture.id())), 
                     ImVec2(imgWidth, imgHeight),
                     ImVec2(0, 0),   // UV0: top-left
                     ImVec2(1, 1));  // UV1: bottom-right
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "No image (tex=%d, w=%d)", 
                          texture.id(), texture.width());
    }
    
    ImGui::End();
}

void StereoViewer::renderInfoPanel() {
    ImGui::SetNextWindowPos(ImVec2(10, 640));
    ImGui::SetNextWindowSize(ImVec2(280, 200));
    
    ImGui::Begin("Performance", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    
    // FPS
    ImGui::Text("UI FPS: %.1f", fps_);
    
    ImGui::Separator();
    ImGui::Text("SGBM Timing:");
    
    // Timing bars
    float maxTime = std::max({totalTimeMs_, 200.0f});
    
    ImGui::Text("Total: %.1f ms (%.1f FPS)", totalTimeMs_, 1000.0f / std::max(totalTimeMs_, 0.001f));
    ImGui::ProgressBar(totalTimeMs_ / maxTime, ImVec2(-1, 0));
    
    ImGui::Text("  Cost: %.1f ms", costTimeMs_);
    ImGui::ProgressBar(costTimeMs_ / maxTime, ImVec2(-1, 0));
    
    ImGui::Text("  Aggregation: %.1f ms", aggTimeMs_);
    ImGui::ProgressBar(aggTimeMs_ / maxTime, ImVec2(-1, 0));
    
    ImGui::Text("  Post-process: %.1f ms", postTimeMs_);
    ImGui::ProgressBar(postTimeMs_ / maxTime, ImVec2(-1, 0));
    
    ImGui::End();
}

void StereoViewer::renderProbeOverlay() {
    if (!probeActive_) return;
    
    ImGui::SetNextWindowPos(ImVec2(10, 850));
    ImGui::SetNextWindowSize(ImVec2(280, 80));
    
    ImGui::Begin("Probe", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    
    ImGui::Text("Pixel: (%d, %d)", probeX_, probeY_);
    ImGui::Text("Disparity: %.2f px", probeDisparity_);
    
    if (probeDepth_ > 0 && probeDepth_ < 1000) {
        ImGui::Text("Depth: %.2f m", probeDepth_);
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Depth: invalid");
    }
    
    ImGui::End();
}

void StereoViewer::handleMouseClick(double xpos, double ypos) {
    // This is a simplified version - would need to map screen coords to image coords
    // For now, just show that clicking works
    
    if (disparityRaw_.empty()) return;
    
    // Get window dimensions
    int windowWidth, windowHeight;
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);
    
    // Check if click is in disparity panel area (bottom left)
    int controlPanelWidth = 300;
    int panelWidth = (windowWidth - controlPanelWidth - 20) / 2;
    int panelHeight = (windowHeight - 60) / 2;
    
    float dispPanelX = static_cast<float>(controlPanelWidth + 10);
    float dispPanelY = static_cast<float>(30 + panelHeight);
    
    if (xpos >= dispPanelX && xpos < dispPanelX + panelWidth &&
        ypos >= dispPanelY && ypos < dispPanelY + panelHeight) {
        
        // Map to image coordinates
        float relX = (static_cast<float>(xpos) - dispPanelX) / panelWidth;
        float relY = (static_cast<float>(ypos) - dispPanelY) / panelHeight;
        
        probeX_ = static_cast<int>(relX * disparityRaw_.cols);
        probeY_ = static_cast<int>(relY * disparityRaw_.rows);
        
        // Clamp to valid range
        probeX_ = std::max(0, std::min(probeX_, disparityRaw_.cols - 1));
        probeY_ = std::max(0, std::min(probeY_, disparityRaw_.rows - 1));
        
        // Get disparity value
        int16_t d = disparityRaw_.at<int16_t>(probeY_, probeX_);
        probeDisparity_ = static_cast<float>(d) / 16.0f;
        
        // Calculate depth
        if (probeDisparity_ > 0 && camParams_.focalLength > 0) {
            probeDepth_ = static_cast<float>(camParams_.focalLength * camParams_.baseline / probeDisparity_);
        } else {
            probeDepth_ = -1;
        }
        
        probeActive_ = true;
    }
}

} // namespace stereo
