/**
 * sgbm_kernels.cu - CUDA kernels for SGBM stereo matching
 * 
 * Implements:
 * 1. Image prefiltering (Sobel-x + clip for illumination normalization)
 * 2. Cost computation (SAD mean or Census Hamming, stored as uint8)
 * 3. SGM cost aggregation (4/8 directions, costScale amplification, adaptive P2)
 * 4. Disparity selection (WTA with uniqueness check)
 * 5. Post-processing (LR consistency, median filter)
 *
 * Cost-to-penalty ratio fix:
 *   Costs are stored as uint8 [0,255] to save VRAM, but aggregation multiplies
 *   each cost read by `costScale` so that the effective cost range is large enough
 *   relative to P1/P2 penalties. Without this, smoothness completely dominates
 *   the data term, producing over-smoothed disparity maps.
 */

#include "stereo/common.h"
#include "stereo/sgbm_limits.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace stereo {

#define TILE_WIDTH 32
#define TILE_HEIGHT 16

// ============================================================================
// PREFILTER KERNEL
// ============================================================================

/**
 * Sobel-x prefilter: computes horizontal gradient and clips to [-cap, cap],
 * then shifts to [0, 2*cap]. This bounds the per-pixel SAD cost contribution
 * and normalizes against illumination bias (matching gradients, not raw intensity).
 */
__global__ void prefilterXSobel(
    const unsigned char* __restrict__ img,
    unsigned char* __restrict__ filtered,
    int width,
    int height,
    int cap
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int val;
    if (x > 0 && x < width - 1) {
        val = (int)img[y * width + x + 1] - (int)img[y * width + x - 1];
    } else if (x == 0) {
        val = (int)img[y * width + 1] - (int)img[y * width + 0];
    } else {
        val = (int)img[y * width + x] - (int)img[y * width + x - 1];
    }
    val = min(cap, max(-cap, val));
    filtered[y * width + x] = static_cast<unsigned char>(val + cap);
}

// ============================================================================
// COST COMPUTATION KERNELS
// ============================================================================

/**
 * SAD cost computation with shared memory optimization.
 * Stores MEAN (sum/count) as uint8. The aggregation kernels multiply by
 * costScale to recover a properly-scaled effective cost.
 */
__global__ void computeCostSAD_shared(
    const unsigned char* __restrict__ left,
    const unsigned char* __restrict__ right,
    unsigned char* __restrict__ costVolume,
    int width,
    int height,
    int maxDisparity,
    int minDisparity,
    int blockRadius
) {
    extern __shared__ unsigned char sharedMem[];
    
    const int padRadius = blockRadius;
    const int leftTileW = TILE_WIDTH + 2 * padRadius;
    const int leftTileH = TILE_HEIGHT + 2 * padRadius;
    const int rightTileW = TILE_WIDTH + 2 * padRadius + minDisparity + maxDisparity + padRadius;
    const int rightTileH = TILE_HEIGHT + 2 * padRadius;
    
    unsigned char* leftTile = sharedMem;
    unsigned char* rightTile = sharedMem + leftTileW * leftTileH;
    
    int gx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int gy = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    int tileOriginX = blockIdx.x * TILE_WIDTH - padRadius;
    int tileOriginY = blockIdx.y * TILE_HEIGHT - padRadius;
    
    int numLeftPixels = leftTileW * leftTileH;
    int threadsPerBlock = TILE_WIDTH * TILE_HEIGHT;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    
    for (int i = tid; i < numLeftPixels; i += threadsPerBlock) {
        int ty = i / leftTileW;
        int tx = i % leftTileW;
        int srcX = tileOriginX + tx;
        int srcY = tileOriginY + ty;
        
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            leftTile[ty * leftTileW + tx] = left[srcY * width + srcX];
        } else {
            leftTile[ty * leftTileW + tx] = 0;
        }
    }
    
    int rightOriginX = tileOriginX - (minDisparity + maxDisparity - 1);
    int numRightPixels = rightTileW * rightTileH;
    
    for (int i = tid; i < numRightPixels; i += threadsPerBlock) {
        int ty = i / rightTileW;
        int tx = i % rightTileW;
        int srcX = rightOriginX + tx;
        int srcY = tileOriginY + ty;
        
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            rightTile[ty * rightTileW + tx] = right[srcY * width + srcX];
        } else {
            rightTile[ty * rightTileW + tx] = 0;
        }
    }
    
    __syncthreads();
    
    if (gx >= width || gy >= height) return;
    
    int lx = threadIdx.x + padRadius;
    int ly = threadIdx.y + padRadius;
    
    for (int d = 0; d < maxDisparity; d++) {
        int sum = 0;
        int count = 0;
        int actualDisp = minDisparity + d;
        
        int rx = lx - d + maxDisparity - 1;
        
        for (int dy = -blockRadius; dy <= blockRadius; dy++) {
            for (int dx = -blockRadius; dx <= blockRadius; dx++) {
                int globalLX = gx + dx;
                int globalRX = gx + dx - actualDisp;
                int globalY = gy + dy;
                
                if (globalLX >= 0 && globalLX < width && 
                    globalRX >= 0 && globalRX < width &&
                    globalY >= 0 && globalY < height) {
                    int rxLocal = rx + dx;
                    if (rxLocal >= 0 && rxLocal < rightTileW) {
                        int leftVal = leftTile[(ly + dy) * leftTileW + (lx + dx)];
                        int rightVal = rightTile[(ly + dy) * rightTileW + rxLocal];
                        sum += abs(leftVal - rightVal);
                        count++;
                    }
                }
            }
        }
        
        int idx = (gy * width + gx) * maxDisparity + d;
        costVolume[idx] = static_cast<unsigned char>((count > 0) ? (sum / count) : 255);
    }
}

/**
 * SAD cost computation - naive version (fallback when shared memory is insufficient).
 */
__global__ void computeCostSAD_naive(
    const unsigned char* __restrict__ left,
    const unsigned char* __restrict__ right,
    unsigned char* __restrict__ costVolume,
    int width,
    int height,
    int maxDisparity,
    int minDisparity,
    int blockRadius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int d = 0; d < maxDisparity; d++) {
        int sum = 0;
        int count = 0;
        int actualDisp = minDisparity + d;
        
        for (int dy = -blockRadius; dy <= blockRadius; dy++) {
            for (int dx = -blockRadius; dx <= blockRadius; dx++) {
                int lx = x + dx;
                int ly = y + dy;
                int rx = x + dx - actualDisp;
                int ry = y + dy;
                
                if (lx >= 0 && lx < width && ly >= 0 && ly < height &&
                    rx >= 0 && rx < width && ry >= 0 && ry < height) {
                    int leftVal = left[ly * width + lx];
                    int rightVal = right[ry * width + rx];
                    sum += abs(leftVal - rightVal);
                    count++;
                }
            }
        }
        
        int idx = (y * width + x) * maxDisparity + d;
        costVolume[idx] = static_cast<unsigned char>((count > 0) ? (sum / count) : 255);
    }
}

/**
 * 5x5 sparse census transform (24 bits: each neighbor vs center, row-major skipping center).
 */
__global__ void buildCensus5x5(
    const unsigned char* __restrict__ img,
    uint32_t* __restrict__ census,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) {
        census[y * width + x] = 0u;
        return;
    }

    unsigned char c = img[y * width + x];
    uint32_t bits = 0u;
    int bit = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (img[(y + dy) * width + (x + dx)] >= c) {
                bits |= (1u << bit);
            }
            ++bit;
        }
    }
    census[y * width + x] = bits;
}

/**
 * Census matching cost: Hamming distance scaled to 0..255 for uint8 cost volume.
 */
__global__ void computeCostCensus_naive(
    const uint32_t* __restrict__ leftCensus,
    const uint32_t* __restrict__ rightCensus,
    unsigned char* __restrict__ costVolume,
    int width,
    int height,
    int maxDisparity,
    int minDisparity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const bool leftOk = (x >= 2 && x < width - 2 && y >= 2 && y < height - 2);
    const uint32_t cL = leftCensus[y * width + x];
    const int baseIdx = (y * width + x) * maxDisparity;

    if (!leftOk) {
        for (int d = 0; d < maxDisparity; ++d) {
            costVolume[baseIdx + d] = 255;
        }
        return;
    }

    for (int d = 0; d < maxDisparity; ++d) {
        const int actualDisp = minDisparity + d;
        const int rx = x - actualDisp;
        const bool rightOk = (rx >= 2 && rx < width - 2 && y >= 2 && y < height - 2);
        if (!rightOk) {
            costVolume[baseIdx + d] = 255;
            continue;
        }
        const uint32_t cR = rightCensus[y * width + rx];
        const int h = __popc(cL ^ cR);
        costVolume[baseIdx + d] = static_cast<unsigned char>((h * 255 + 12) / 24);
    }
}

// ============================================================================
// SGM COST AGGREGATION KERNELS (with costScale amplification + adaptive P2)
// ============================================================================

__device__ __forceinline__ uint16_t computeMinCost(const uint16_t* L, int maxDisparity) {
    uint16_t minVal = UINT16_MAX;
    for (int d = 0; d < maxDisparity; d++) {
        if (L[d] < minVal) minVal = L[d];
    }
    return minVal;
}

/**
 * Reduce P2 at intensity edges to allow disparity discontinuities at
 * object boundaries.  P2_effective = max(P1+1, P2 / |gradient|).
 */
__device__ __forceinline__ int adaptiveP2(int P1, int P2, int imgCurr, int imgPrev) {
    int gradient = abs(imgCurr - imgPrev);
    // Smoothly reduce P2 only at strong edges; use threshold of 8 so that
    // minor texture doesn't destroy smoothness.  Result: P2 at gradient<8,
    // P2/2 at gradient=16, P2/4 at gradient=32, then clamped to P1+1.
    return max(P1 + 1, P2 * 8 / max(8, gradient));
}

/**
 * SGM aggregation: left to right
 */
__global__ void aggregateCostHorizontalLR(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    
    uint16_t L_prev[kSgbmMaxDisparityRange];
    uint16_t L_curr[kSgbmMaxDisparityRange];
    
    int baseIdx = y * width * maxDisparity;
    for (int d = 0; d < maxDisparity; d++) {
        L_prev[d] = static_cast<uint16_t>(costVolume[baseIdx + d]) * costScale;
        aggCost[baseIdx + d] = L_prev[d];
    }
    
    for (int x = 1; x < width; x++) {
        int currBase = (y * width + x) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[y * width + x - 1]);
        uint16_t minPrev = computeMinCost(L_prev, maxDisparity);
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_prev[d];
            uint16_t cost1 = (d > 0) ? L_prev[d-1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity-1) ? L_prev[d+1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            L_curr[d] = C + minCost - minPrev;
            
            aggCost[currBase + d] += L_curr[d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            L_prev[d] = L_curr[d];
        }
    }
}

/**
 * SGM aggregation: right to left
 */
__global__ void aggregateCostHorizontalRL(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    
    uint16_t L_prev[kSgbmMaxDisparityRange];
    uint16_t L_curr[kSgbmMaxDisparityRange];
    
    int baseIdx = (y * width + (width - 1)) * maxDisparity;
    for (int d = 0; d < maxDisparity; d++) {
        L_prev[d] = static_cast<uint16_t>(costVolume[baseIdx + d]) * costScale;
        aggCost[baseIdx + d] += L_prev[d];
    }
    
    for (int x = width - 2; x >= 0; x--) {
        int currBase = (y * width + x) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[y * width + x + 1]);
        uint16_t minPrev = computeMinCost(L_prev, maxDisparity);
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_prev[d];
            uint16_t cost1 = (d > 0) ? L_prev[d-1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity-1) ? L_prev[d+1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            L_curr[d] = C + minCost - minPrev;
            
            aggCost[currBase + d] += L_curr[d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            L_prev[d] = L_curr[d];
        }
    }
}

/**
 * SGM aggregation: top to bottom
 */
__global__ void aggregateCostVerticalTB(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    
    uint16_t L_prev[kSgbmMaxDisparityRange];
    uint16_t L_curr[kSgbmMaxDisparityRange];
    
    int baseIdx = x * maxDisparity;
    for (int d = 0; d < maxDisparity; d++) {
        L_prev[d] = static_cast<uint16_t>(costVolume[baseIdx + d]) * costScale;
        aggCost[baseIdx + d] += L_prev[d];
    }
    
    for (int y = 1; y < height; y++) {
        int currBase = (y * width + x) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[(y - 1) * width + x]);
        uint16_t minPrev = computeMinCost(L_prev, maxDisparity);
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_prev[d];
            uint16_t cost1 = (d > 0) ? L_prev[d-1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity-1) ? L_prev[d+1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            L_curr[d] = C + minCost - minPrev;
            
            aggCost[currBase + d] += L_curr[d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            L_prev[d] = L_curr[d];
        }
    }
}

/**
 * SGM aggregation: bottom to top
 */
__global__ void aggregateCostVerticalBT(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    
    uint16_t L_prev[kSgbmMaxDisparityRange];
    uint16_t L_curr[kSgbmMaxDisparityRange];
    
    int baseIdx = ((height - 1) * width + x) * maxDisparity;
    for (int d = 0; d < maxDisparity; d++) {
        L_prev[d] = static_cast<uint16_t>(costVolume[baseIdx + d]) * costScale;
        aggCost[baseIdx + d] += L_prev[d];
    }
    
    for (int y = height - 2; y >= 0; y--) {
        int currBase = (y * width + x) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[(y + 1) * width + x]);
        uint16_t minPrev = computeMinCost(L_prev, maxDisparity);
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_prev[d];
            uint16_t cost1 = (d > 0) ? L_prev[d-1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity-1) ? L_prev[d+1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            L_curr[d] = C + minCost - minPrev;
            
            aggCost[currBase + d] += L_curr[d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            L_prev[d] = L_curr[d];
        }
    }
}

/**
 * SGM aggregation: top-left to bottom-right diagonal
 */
__global__ void aggregateCostDiagonalTLBR(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    uint16_t* __restrict__ L_diag,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale,
    int diagIdx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = tid;
    int x = diagIdx - y;
    
    if (y < 0 || y >= height || x < 0 || x >= width) return;
    
    int currBase = (y * width + x) * maxDisparity;
    int diagBase = currBase;
    
    if (x == 0 || y == 0) {
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            L_diag[diagBase + d] = C;
            aggCost[currBase + d] += C;
        }
    } else {
        int prevBase = ((y - 1) * width + (x - 1)) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[(y - 1) * width + (x - 1)]);
        
        uint16_t minPrev = UINT16_MAX;
        for (int d = 0; d < maxDisparity; d++) {
            if (L_diag[prevBase + d] < minPrev) minPrev = L_diag[prevBase + d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_diag[prevBase + d];
            uint16_t cost1 = (d > 0) ? L_diag[prevBase + d - 1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity - 1) ? L_diag[prevBase + d + 1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            uint16_t L = C + minCost - minPrev;
            
            L_diag[diagBase + d] = L;
            aggCost[currBase + d] += L;
        }
    }
}

/**
 * SGM aggregation: bottom-right to top-left diagonal
 */
__global__ void aggregateCostDiagonalBRTL(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    uint16_t* __restrict__ L_diag,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale,
    int diagIdx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = height - 1 - tid;
    int x = width - 1 - (diagIdx - (height - 1 - y));
    
    if (y < 0 || y >= height || x < 0 || x >= width) return;
    
    int currBase = (y * width + x) * maxDisparity;
    int diagBase = currBase;
    
    if (x == width - 1 || y == height - 1) {
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            L_diag[diagBase + d] = C;
            aggCost[currBase + d] += C;
        }
    } else {
        int prevBase = ((y + 1) * width + (x + 1)) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[(y + 1) * width + (x + 1)]);
        
        uint16_t minPrev = UINT16_MAX;
        for (int d = 0; d < maxDisparity; d++) {
            if (L_diag[prevBase + d] < minPrev) minPrev = L_diag[prevBase + d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_diag[prevBase + d];
            uint16_t cost1 = (d > 0) ? L_diag[prevBase + d - 1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity - 1) ? L_diag[prevBase + d + 1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            uint16_t L = C + minCost - minPrev;
            
            L_diag[diagBase + d] = L;
            aggCost[currBase + d] += L;
        }
    }
}

/**
 * SGM aggregation: top-right to bottom-left diagonal
 */
__global__ void aggregateCostDiagonalTRBL(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    uint16_t* __restrict__ L_diag,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale,
    int diagIdx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = tid;
    int x = width - 1 - (diagIdx - y);
    
    if (y < 0 || y >= height || x < 0 || x >= width) return;
    
    int currBase = (y * width + x) * maxDisparity;
    int diagBase = currBase;
    
    if (x == width - 1 || y == 0) {
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            L_diag[diagBase + d] = C;
            aggCost[currBase + d] += C;
        }
    } else {
        int prevBase = ((y - 1) * width + (x + 1)) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[(y - 1) * width + (x + 1)]);
        
        uint16_t minPrev = UINT16_MAX;
        for (int d = 0; d < maxDisparity; d++) {
            if (L_diag[prevBase + d] < minPrev) minPrev = L_diag[prevBase + d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_diag[prevBase + d];
            uint16_t cost1 = (d > 0) ? L_diag[prevBase + d - 1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity - 1) ? L_diag[prevBase + d + 1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            uint16_t L = C + minCost - minPrev;
            
            L_diag[diagBase + d] = L;
            aggCost[currBase + d] += L;
        }
    }
}

/**
 * SGM aggregation: bottom-left to top-right diagonal
 */
__global__ void aggregateCostDiagonalBLTR(
    const unsigned char* __restrict__ costVolume,
    uint16_t* __restrict__ aggCost,
    uint16_t* __restrict__ L_diag,
    const unsigned char* __restrict__ img,
    int width,
    int height,
    int maxDisparity,
    int P1,
    int P2,
    int costScale,
    int diagIdx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = height - 1 - tid;
    int x = diagIdx - (height - 1 - y);
    
    if (y < 0 || y >= height || x < 0 || x >= width) return;
    
    int currBase = (y * width + x) * maxDisparity;
    int diagBase = currBase;
    
    if (x == 0 || y == height - 1) {
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            L_diag[diagBase + d] = C;
            aggCost[currBase + d] += C;
        }
    } else {
        int prevBase = ((y + 1) * width + (x - 1)) * maxDisparity;
        int P2adj = adaptiveP2(P1, P2, img[y * width + x], img[(y + 1) * width + (x - 1)]);
        
        uint16_t minPrev = UINT16_MAX;
        for (int d = 0; d < maxDisparity; d++) {
            if (L_diag[prevBase + d] < minPrev) minPrev = L_diag[prevBase + d];
        }
        
        for (int d = 0; d < maxDisparity; d++) {
            uint16_t C = static_cast<uint16_t>(costVolume[currBase + d]) * costScale;
            
            uint16_t cost0 = L_diag[prevBase + d];
            uint16_t cost1 = (d > 0) ? L_diag[prevBase + d - 1] + P1 : UINT16_MAX;
            uint16_t cost2 = (d < maxDisparity - 1) ? L_diag[prevBase + d + 1] + P1 : UINT16_MAX;
            uint16_t cost3 = minPrev + P2adj;
            
            uint16_t minCost = min(min(cost0, cost1), min(cost2, cost3));
            uint16_t L = C + minCost - minPrev;
            
            L_diag[diagBase + d] = L;
            aggCost[currBase + d] += L;
        }
    }
}

// ============================================================================
// DISPARITY SELECTION KERNELS
// ============================================================================

__global__ void selectDisparityWTA(
    const uint16_t* __restrict__ aggregatedCost,
    int16_t* __restrict__ disparity,
    int width,
    int height,
    int maxDisparity,
    int minDisparity,
    int uniquenessRatio
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int baseIdx = (y * width + x) * maxDisparity;
    
    uint16_t minCost = UINT16_MAX;
    uint16_t secondMinCost = UINT16_MAX;
    int bestD = 0;
    
    for (int d = 0; d < maxDisparity; d++) {
        uint16_t cost = aggregatedCost[baseIdx + d];
        if (cost < minCost) {
            secondMinCost = minCost;
            minCost = cost;
            bestD = d;
        } else if (cost < secondMinCost) {
            secondMinCost = cost;
        }
    }
    
    if (uniquenessRatio > 0 && secondMinCost != UINT16_MAX) {
        const uint32_t best = static_cast<uint32_t>(minCost);
        const uint32_t second = static_cast<uint32_t>(secondMinCost);
        const uint32_t thresh = (best * static_cast<uint32_t>(uniquenessRatio)) / 100u;
        if (second <= best + thresh) {
            disparity[y * width + x] = -1;
            return;
        }
    }
    
    float subpixelD = static_cast<float>(bestD);
    
    if (bestD > 0 && bestD < maxDisparity - 1) {
        float c0 = static_cast<float>(aggregatedCost[baseIdx + bestD - 1]);
        float c1 = static_cast<float>(aggregatedCost[baseIdx + bestD]);
        float c2 = static_cast<float>(aggregatedCost[baseIdx + bestD + 1]);
        
        float denom = 2.0f * (c0 + c2 - 2.0f * c1);
        if (fabsf(denom) > 1e-6f) {
            float delta = (c0 - c2) / denom;
            delta = fmaxf(-1.0f, fminf(1.0f, delta));
            subpixelD += delta;
        }
    }
    
    disparity[y * width + x] = static_cast<int16_t>((subpixelD + minDisparity) * 16.0f);
}

__global__ void selectDisparityRight(
    const uint16_t* __restrict__ aggregatedCost,
    int16_t* __restrict__ disparityR,
    int width,
    int height,
    int maxDisparity,
    int minDisparity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint16_t minCost = UINT16_MAX;
    int bestD = 0;
    
    for (int d = 0; d < maxDisparity; d++) {
        int leftX = x + minDisparity + d;
        if (leftX >= width) break;
        
        int baseIdx = (y * width + leftX) * maxDisparity + d;
        uint16_t cost = aggregatedCost[baseIdx];
        
        if (cost < minCost) {
            minCost = cost;
            bestD = d;
        }
    }
    
    float subpixelD = static_cast<float>(bestD);
    
    if (bestD > 0 && bestD < maxDisparity - 1) {
        int leftX = x + minDisparity + bestD;
        if (leftX < width) {
            int baseIdx = (y * width + leftX) * maxDisparity;
            float c0 = static_cast<float>(aggregatedCost[baseIdx + bestD - 1]);
            float c1 = static_cast<float>(aggregatedCost[baseIdx + bestD]);
            float c2 = static_cast<float>(aggregatedCost[baseIdx + bestD + 1]);
            
            float denom = 2.0f * (c0 + c2 - 2.0f * c1);
            if (fabsf(denom) > 1e-6f) {
                float delta = (c0 - c2) / denom;
                delta = fmaxf(-1.0f, fminf(1.0f, delta));
                subpixelD += delta;
            }
        }
    }
    
    disparityR[y * width + x] = static_cast<int16_t>((subpixelD + minDisparity) * 16.0f);
}

// ============================================================================
// POST-PROCESSING KERNELS
// ============================================================================

__global__ void consistencyCheck(
    int16_t* __restrict__ disparityL,
    const int16_t* __restrict__ disparityR,
    int width,
    int height,
    int maxDiff
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int16_t dL = disparityL[y * width + x];
    if (dL <= 0) return;
    
    int d = dL / 16;
    int xR = x - d;
    
    if (xR < 0 || xR >= width) {
        disparityL[y * width + x] = -1;
        return;
    }
    
    int16_t dR = disparityR[y * width + xR];
    
    if (dR <= 0 || abs(dL - dR) > maxDiff) {
        disparityL[y * width + x] = -1;
    }
}

__global__ void medianFilter3x3(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int16_t values[9];
    int count = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int16_t val = input[ny * width + nx];
                if (val > 0) {
                    values[count++] = val;
                }
            }
        }
    }
    
    if (count == 0) {
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (values[j] < values[i]) {
                int16_t tmp = values[i];
                values[i] = values[j];
                values[j] = tmp;
            }
        }
    }
    
    output[y * width + x] = values[count / 2];
}

__global__ void fillSmallHoles(
    int16_t* __restrict__ disparity,
    const int16_t* __restrict__ original,
    int width,
    int height,
    int maxHoleSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int16_t val = original[y * width + x];
    if (val > 0) {
        disparity[y * width + x] = val;
        return;
    }
    
    int16_t sum = 0;
    int count = 0;
    
    for (int dy = -maxHoleSize; dy <= maxHoleSize; dy++) {
        for (int dx = -maxHoleSize; dx <= maxHoleSize; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int16_t nval = original[ny * width + nx];
                if (nval > 0) {
                    sum += nval;
                    count++;
                }
            }
        }
    }
    
    if (count > 0) {
        disparity[y * width + x] = sum / count;
    } else {
        disparity[y * width + x] = -1;
    }
}

__global__ void speckleFilter(
    int16_t* __restrict__ disparity,
    int width,
    int height,
    int maxSpeckleSize,
    int maxDiff
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int16_t centerVal = disparity[y * width + x];
    if (centerVal <= 0) return;
    
    int similarCount = 0;
    const int radius = 2;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int16_t nval = disparity[ny * width + nx];
                if (nval > 0 && abs(centerVal - nval) <= maxDiff) {
                    similarCount++;
                }
            }
        }
    }
    
    if (similarCount < maxSpeckleSize) {
        disparity[y * width + x] = -1;
    }
}

} // namespace stereo
