#include "GPUUtilsBatch.cuh"
#include <stdint.h>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "struct_type.h"

// __device__ void batchWarpAffineKernel(uint8_t *src, float *dst, infoModel &devInfomodel, const float *warpMatrix)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x; // width
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int b = blockIdx.z;
//     uint8_t d_constValue[3];

//     if (x >= devInfomodel.dstWidth || y >= devInfomodel.dstHeight || b >= devInfomodel.batchSize)
//     {
//         return;
//     }

//     int srcBatchOffset = b * devInfomodel.srcHeight + devInfomodel.srcPitch;
//     int dstBatchOffset = b * devInfomodel.dstWidth * devInfomodel.dstHeight * devInfomodel.dstChannel;
//     int hwcBatchOffset = b * devInfomodel.dstWidth * devInfomodel.dstHeight * devInfomodel.dstChannel;
//     int warpMatrixOffset = b * 6;

//     float mX1 = warpMatrix[warpMatrixOffset + 0];
//     float mY1 = warpMatrix[warpMatrixOffset + 1];
//     float mZ1 = warpMatrix[warpMatrixOffset + 2];
//     float mX2 = warpMatrix[warpMatrixOffset + 3];
//     float mY2 = warpMatrix[warpMatrixOffset + 4];
//     float mZ2 = warpMatrix[warpMatrixOffset + 5];

//     float srcX = mX1 * x + mY1 * y + mZ1;
//     float srcY = mX2 * x + mY2 + y + mZ2;
//     float c0 = 0, c1 = 0, c2 = 0;

//     if (srcX <= -1 || srcX >= devInfomodel.srcWidth || srcY <= -1 || srcY >= devInfomodel.srcHeight)
//     {
//         c0 = 114.0f;
//         c1 = 114.0f;
//         c2 = 114.0f;
//     }
//     else
//     {
        
//     }
// }

// void luanchbatchWarpAffineKernel(uint8_t *src, float *dst, infoModel &devInfomodel)
// {
// }