#pragma once

/**
 * common.h - Core utilities, error handling, and CUDA helpers
 * 
 * This file provides:
 * 1. CUDA error checking macros
 * 2. Exception classes for error handling
 * 3. Common type definitions
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace stereo {

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

/**
 * Exception thrown when a CUDA operation fails.
 * 
 * We wrap CUDA errors in C++ exceptions because:
 * - CUDA functions return error codes (C-style), not exceptions
 * - Exceptions provide stack traces and can propagate up
 * - Easier to handle errors at appropriate level
 */
class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(formatError(err, file, line))
        , error_code(err)
    {}
    
    cudaError_t error_code;
    
private:
    static std::string formatError(cudaError_t err, const char* file, int line) {
        std::ostringstream ss;
        ss << "CUDA error at " << file << ":" << line << "\n"
           << "  Error code: " << static_cast<int>(err) << "\n"
           << "  Error string: " << cudaGetErrorString(err);
        return ss.str();
    }
};

/**
 * CUDA_CHECK macro - wraps every CUDA call to check for errors
 * 
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 * 
 * If the call fails, it throws CudaException with file/line info.
 * The do-while(0) is a C/C++ idiom that makes the macro behave like a statement.
 */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            throw stereo::CudaException(err, __FILE__, __LINE__);          \
        }                                                                   \
    } while (0)

/**
 * Check for kernel launch errors
 * 
 * CUDA kernel launches are asynchronous and don't return errors immediately.
 * After launching a kernel, call this to check if it failed.
 * 
 * Usage:
 *   myKernel<<<blocks, threads>>>(...);
 *   CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                          \
            throw stereo::CudaException(err, __FILE__, __LINE__);          \
        }                                                                   \
    } while (0)

/**
 * Synchronize and check - useful for debugging
 * Forces GPU to finish all work, then checks for errors.
 * Slower but catches errors exactly where they occur.
 */
#define CUDA_SYNC_CHECK()                                                   \
    do {                                                                    \
        CUDA_CHECK(cudaDeviceSynchronize());                               \
        CUDA_CHECK_KERNEL();                                                \
    } while (0)


// ============================================================================
// STEREO-SPECIFIC EXCEPTIONS
// ============================================================================

/**
 * General stereo pipeline exception
 */
class StereoException : public std::runtime_error {
public:
    explicit StereoException(const std::string& msg) 
        : std::runtime_error(msg) 
    {}
};

/**
 * Calibration-specific errors
 */
class CalibrationException : public StereoException {
public:
    explicit CalibrationException(const std::string& msg)
        : StereoException("Calibration error: " + msg)
    {}
};


// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Calculate number of thread blocks needed to cover N elements
 * 
 * This is a common pattern in CUDA:
 *   dim3 blocks = divUp(width, 32);
 *   myKernel<<<blocks, 32>>>(...);
 * 
 * Example: divUp(1000, 256) = 4 (4 blocks × 256 threads = 1024 ≥ 1000)
 */
inline int divUp(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * 2D version for image processing
 */
inline dim3 divUp(int width, int height, int blockX, int blockY) {
    return dim3(divUp(width, blockX), divUp(height, blockY));
}

/**
 * Print GPU device information (useful for debugging)
 * Returns false if CUDA is not available
 */
inline bool printDeviceInfo() {
    int deviceId;
    cudaError_t err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
        printf("CUDA not available: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, deviceId);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    printf("GPU Device %d: %s\n", deviceId, props.name);
    printf("  Compute capability: %d.%d\n", props.major, props.minor);
    printf("  Total memory: %.2f GB\n", props.totalGlobalMem / 1e9);
    printf("  Shared memory per block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("  Warp size: %d\n", props.warpSize);
    return true;
}

} // namespace stereo
