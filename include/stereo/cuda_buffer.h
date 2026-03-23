#pragma once

/**
 * cuda_buffer.h - RAII wrapper for GPU memory
 * 
 * Why do we need this?
 * 
 * Raw CUDA memory management looks like this:
 *   float* d_ptr;
 *   cudaMalloc(&d_ptr, size * sizeof(float));
 *   // ... use d_ptr ...
 *   cudaFree(d_ptr);  // Easy to forget!
 * 
 * Problems:
 * 1. If an exception is thrown, cudaFree never runs = memory leak
 * 2. Manual tracking of pointer validity is error-prone
 * 3. No type safety (void* everywhere)
 * 
 * CudaBuffer solves this with RAII:
 *   CudaBuffer<float> buffer(size);  // Allocates
 *   // ... use buffer.data() ...
 *   // Automatically freed when buffer goes out of scope
 */

#include "common.h"
#include <cstring>

namespace stereo {

/**
 * RAII wrapper for device (GPU) memory
 * 
 * Usage:
 *   // Allocate 1000 floats on GPU
 *   CudaBuffer<float> d_data(1000);
 *   
 *   // Get raw pointer for kernel calls
 *   myKernel<<<...>>>(d_data.data(), d_data.size());
 *   
 *   // Copy from CPU to GPU
 *   std::vector<float> h_data(1000);
 *   d_data.copyFrom(h_data.data());
 *   
 *   // Copy from GPU to CPU
 *   d_data.copyTo(h_data.data());
 *   
 *   // Memory automatically freed when d_data goes out of scope
 */
template<typename T>
class CudaBuffer {
public:
    // Default constructor - creates empty buffer
    CudaBuffer() : data_(nullptr), size_(0) {}
    
    // Allocate buffer of given size (number of elements, not bytes!)
    explicit CudaBuffer(size_t size) : data_(nullptr), size_(0) {
        allocate(size);
    }
    
    // Destructor - free GPU memory
    ~CudaBuffer() {
        free();
    }
    
    // ========================================================================
    // MOVE SEMANTICS
    // ========================================================================
    // We allow moving but not copying (copying GPU memory is expensive)
    
    // Move constructor - "steal" the pointer from other
    CudaBuffer(CudaBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) 
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations (use explicit copyFrom if you really need to copy)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // ========================================================================
    // MEMORY MANAGEMENT
    // ========================================================================
    
    /**
     * Allocate GPU memory
     * @param size Number of elements (not bytes)
     */
    void allocate(size_t size) {
        if (size == size_ && data_ != nullptr) {
            return;  // Already allocated with correct size
        }
        
        free();  // Free existing memory first
        
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size * sizeof(T)));
            size_ = size;
        }
    }
    
    /**
     * Free GPU memory
     */
    void free() {
        if (data_ != nullptr) {
            cudaFree(data_);  // Note: we don't check errors on free
            data_ = nullptr;
            size_ = 0;
        }
    }
    
    /**
     * Resize buffer (reallocates if size changed)
     */
    void resize(size_t newSize) {
        if (newSize != size_) {
            allocate(newSize);
        }
    }
    
    // ========================================================================
    // DATA TRANSFER
    // ========================================================================
    
    /**
     * Copy data from CPU (host) to GPU (device)
     * @param hostPtr Pointer to CPU memory
     * @param count Number of elements to copy (default: entire buffer)
     */
    void copyFrom(const T* hostPtr, size_t count = 0) {
        if (count == 0) count = size_;
        if (count > size_) {
            throw StereoException("CudaBuffer::copyFrom: count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpy(data_, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    /**
     * Copy data from GPU (device) to CPU (host)
     * @param hostPtr Pointer to CPU memory (must be pre-allocated!)
     * @param count Number of elements to copy (default: entire buffer)
     */
    void copyTo(T* hostPtr, size_t count = 0) const {
        if (count == 0) count = size_;
        if (count > size_) {
            throw StereoException("CudaBuffer::copyTo: count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpy(hostPtr, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    /**
     * Copy from another GPU buffer (device to device)
     */
    void copyFromDevice(const CudaBuffer<T>& other) {
        if (other.size() > size_) {
            throw StereoException("CudaBuffer::copyFromDevice: source larger than destination");
        }
        CUDA_CHECK(cudaMemcpy(data_, other.data(), other.size() * sizeof(T), 
                              cudaMemcpyDeviceToDevice));
    }
    
    /**
     * Set all bytes to zero
     */
    void zero() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
        }
    }
    
    // ========================================================================
    // ACCESSORS
    // ========================================================================
    
    // Get raw device pointer (for kernel calls)
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    // Number of elements
    size_t size() const { return size_; }
    
    // Size in bytes
    size_t bytes() const { return size_ * sizeof(T); }
    
    // Check if allocated
    bool empty() const { return data_ == nullptr || size_ == 0; }
    
    // Implicit conversion to raw pointer (convenient but use carefully)
    operator T*() { return data_; }
    operator const T*() const { return data_; }
    
private:
    T* data_;
    size_t size_;
};

/**
 * Helper to allocate a 2D buffer (for images)
 * Calculates size as width * height * channels
 */
template<typename T>
CudaBuffer<T> make2DBuffer(int width, int height, int channels = 1) {
    return CudaBuffer<T>(static_cast<size_t>(width) * height * channels);
}

} // namespace stereo
