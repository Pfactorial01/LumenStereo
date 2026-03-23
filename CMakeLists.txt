cmake_minimum_required(VERSION 3.18)

# ============================================================================
# PROJECT DEFINITION
# ============================================================================
# We enable CUDA as a first-class language (not just a separate toolkit)
# This lets us use .cu files directly and CMake handles nvcc automatically
project(LumenStereo LANGUAGES CXX CUDA)

# C++17 for modern features (std::optional, structured bindings, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard should match C++ standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# BUILD OPTIONS
# ============================================================================
# These can be toggled with: cmake -DBUILD_TESTS=ON ..
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_VIEWER "Build interactive viewer" ON)

# ============================================================================
# COMPILER FLAGS
# ============================================================================
# Debug vs Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # -O3: Maximum optimization
    # -use_fast_math: Faster but slightly less precise math (fine for graphics)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    # Debug: Enable device-side debugging, line info for profilers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
endif()

# Suppress annoying CUDA warnings about deprecated GPU architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# ============================================================================
# CUDA ARCHITECTURE
# ============================================================================
# This tells nvcc which GPU generations to compile for
# Common architectures:
#   60 = Pascal (GTX 1080)
#   70 = Volta (V100)
#   75 = Turing (RTX 2080)
#   80 = Ampere (RTX 3080, A100)
#   86 = Ampere (RTX 3090)
#   89 = Ada Lovelace (RTX 4090)
#   90 = Hopper (H100)
# "native" auto-detects your GPU (CMake 3.24+)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ============================================================================
# FIND DEPENDENCIES
# ============================================================================
# find_package() searches for installed libraries and sets variables like
# OpenCV_INCLUDE_DIRS, OpenCV_LIBS, etc.

# OpenCV: We use it for calibration math and image I/O
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs highgui)

# OpenGL: For rendering in the viewer
find_package(OpenGL REQUIRED)

# GLFW: Window creation and input
find_package(glfw3 REQUIRED)

# yaml-cpp: Configuration file parsing
find_package(yaml-cpp REQUIRED)

# CUDAToolkit: Provides CUDA include directories for C++ files
# This is needed because our headers include <cuda_runtime.h>
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# FETCH EXTERNAL DEPENDENCIES (downloaded at configure time)
# ============================================================================
include(FetchContent)

# Dear ImGui: Immediate-mode GUI library
# We fetch it from GitHub so users don't need to install it
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_MakeAvailable(imgui)

# ============================================================================
# IMGUI LIBRARY
# ============================================================================
# ImGui doesn't have a CMakeLists.txt, so we create a library manually
# We include the OpenGL3+GLFW backend files
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# ============================================================================
# CORE STEREO LIBRARY
# ============================================================================
# This is our main library containing all stereo vision algorithms
add_library(stereo_core STATIC
    # Source files (we'll add more as we implement them)
    src/calibration.cpp
    src/rectification.cu
    src/sgbm_kernels.cu
    src/sgbm_gpu.cu
    src/depth_map.cu
    src/point_cloud.cpp
    src/config.cpp
)

# Where to find our header files
# We include CUDA headers as PUBLIC so any target linking to stereo_core
# can also find cuda_runtime.h
target_include_directories(stereo_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link dependencies to our library
# CUDA::cudart provides the CUDA runtime library
target_link_libraries(stereo_core PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    CUDA::cudart
)

# Enable separable compilation for CUDA
# This allows CUDA files to call functions defined in other CUDA files
set_target_properties(stereo_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# INTERACTIVE VIEWER APPLICATION
# ============================================================================
if(BUILD_VIEWER)
    add_executable(stereo_viewer
        viewer/main.cpp
        viewer/stereo_viewer.cpp
    )
    
    target_link_libraries(stereo_viewer PRIVATE
        stereo_core
        imgui
        OpenGL::GL
        glfw
    )
    
    # CUDA-OpenGL interop requires linking against CUDA runtime
    set_target_properties(stereo_viewer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================================
# CALIBRATION TOOL
# ============================================================================
add_executable(calibration_tool
    tools/calibration_tool.cpp
)
target_link_libraries(calibration_tool PRIVATE stereo_core)

# ============================================================================
# BENCHMARK TOOL (no GUI required)
# ============================================================================
add_executable(benchmark
    tools/benchmark.cpp
)
target_link_libraries(benchmark PRIVATE stereo_core)

# ============================================================================
# DIAGNOSTIC TOOL
# ============================================================================
add_executable(diagnose
    tools/diagnose.cpp
)
target_link_libraries(diagnose PRIVATE stereo_core)

# ============================================================================
# EVALUATION TOOL (compare against ground truth)
# ============================================================================
add_executable(evaluate
    tools/evaluate.cpp
)
target_link_libraries(evaluate PRIVATE stereo_core)

# ============================================================================
# OPENCV COMPARISON TOOL
# ============================================================================
add_executable(compare_opencv
    tools/compare_opencv.cpp
)
target_link_libraries(compare_opencv PRIVATE stereo_core)

# ============================================================================
# UNIT TESTS
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    
    # Google Test - we build it from source (installed via apt)
    add_subdirectory(/usr/src/gtest ${CMAKE_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)
    
    # Test executable
    add_executable(stereo_tests
        tests/test_main.cpp
        tests/test_rectification.cpp
        tests/test_sgbm.cpp
    )
    
    target_link_libraries(stereo_tests PRIVATE
        stereo_core
        gtest
        gtest_main
    )
    
    # Register with CTest
    add_test(NAME StereoTests COMMAND stereo_tests)
endif()

# ============================================================================
# INSTALLATION (optional)
# ============================================================================
install(TARGETS stereo_core LIBRARY DESTINATION lib)
install(DIRECTORY include/stereo DESTINATION include)
