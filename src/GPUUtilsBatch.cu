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

// struct __align__(16) infoModel
// {
//     int srcWidth;
//     int srcHeight;
//     int srcChannel;
//     int srcPitch;
//     int dstWidth;
//     int dstHeight;
//     int dstChannel;
//     int batchSize;
// };

// __global__ void batchWarpAffineKernel(uint8_t *batchsrc, float *batchdst, float *batchHWC, infoModel *devInfomodel, const float *warpMatrix)
// {
//     int batchIdx = blockIdx.z;
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= devInfomodel->dstWidth || y >= devInfomodel->dstHeight || batchIdx >= devInfomodel->batchSize)
//     {
//         return;
//     }
//     const uint8_t fill = 114;
//     __shared__ float sharedWarpMatrix[6];
//     __shared__ infoModel sharedInfoModel;
//     if (threadIdx.x ==0 && threadIdx.y ==0){
//         #pragma unroll
//         for (int i = 0; i < 6; i++){
//             sharedWarpMatrix[i] = warpMatrix[batchIdx * 6 + i];
//             sharedInfoModel = devInfomodel[batchIdx];
//         }
//     }
//     __syncthreads();
//     const float scale = 1.0f/255.0f;
//     // 計算 source coord
//     float srcX = sharedWarpMatrix[0] * x + sharedWarpMatrix[1] * y + sharedWarpMatrix[2];
//     float srcY = sharedWarpMatrix[3] * x + sharedWarpMatrix[4] * y + sharedWarpMatrix[5];
//     float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

//     if (srcX <= -1.0f || srcY <= -1.0f || srcX >= float(sharedInfoModel.srcWidth) -1.0f ||srcY >= float(sharedInfoModel.srcHeight) -1.0f )
//     {
//         c0 = float(fill);
//         c1 = float(fill);
//         c2 = float(fill);
//     }
//     else{
//         int xLow = floorf(srcX);
//         int yLow = floorf(srcY);
//         int xHigh = xLow + 1;
//         int yHigh = yLow + 1;

//         float lx = srcX - float(xLow);
//         float ly = srcY - float(yLow);
//         float hx = 1.0f - lx;
//         float hy = 1.0f - ly;
//         float w1 = hx * hy; // (xLow,yLow)
//         float w2 = lx * hy; // (xHigh,yLow)
//         float w3 = hx * ly; // (xLow,yHigh)
//         float w4 = lx * ly; // (xHigh,yHigh)
        
//         bool pitchDiv3 = (sharedInfoModel.srcPitch % 3 == 0);
//         if (pitchDiv3)
//         {
//             const uchar3 * src3 = reinterpret_cast<const uchar3 *>(batchsrc);
//             int srcPitch = sharedInfoModel.srcPitch  /3;
//             uchar3 p1 = make_uchar3(fill, fill, fill);
//             uchar3 p2 = p1;
//             uchar3 p3 = p1;
//             uchar3 p4 = p1;
//             if (yLow >= 0 && yLow < sharedInfoModel.srcHeight)
//             {
//                 if (xLow >= 0 && xLow < sharedInfoModel.srcWidth)
//                 {
//                     p1 = src3[batchIdx * sharedInfoModel.srcHeight * srcPitch + yLow * srcPitch + xLow];
//                 }
//                 if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth)
//                 {
//                     p2 = src3[batchIdx * sharedInfoModel.srcHeight * srcPitch + yLow * srcPitch + xHigh];
//                 }
//             }
//             if (yHigh >= 0 && yHigh < sharedInfoModel.srcHeight)
//             {
//                 if (xLow >= 0 && xLow < sharedInfoModel.srcWidth)
//                 {
//                     p3 = src3[batchIdx * sharedInfoModel.srcHeight * srcPitch + yHigh * srcPitch + xLow];
//                 }
//                 if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth)
//                 {
//                     p4 = src3[batchIdx * sharedInfoModel.srcHeight * srcPitch + yHigh * srcPitch + xHigh];
//                 }
//             }
//             c0 = w1 * float(p1.x) + w2 * float(p2.x) + w3 * float(p3.x) + w4 * float(p4.x);
//             c1 = w1 * float(p1.y) + w2 * float(p2.y) + w3 * float(p3.y) + w4 * float(p4.y);
//             c2 = w1 * float(p1.z) + w2 * float(p2.z) + w3 * float(p3.z) + w4 * float(p4.z);
//         }
//         else
//         {
//             const uint8_t *rowBase1 = batchsrc + size_t(yLow) * size_t(sharedInfoModel.srcPitch);
//             const uint8_t *rowBase2 = batchsrc + size_t(yHigh) * size_t(sharedInfoModel.srcPitch);

//             uint8_t v1[3] = {fill, fill, fill};
//             uint8_t v2[3] = {fill, fill, fill};
//             uint8_t v3[3] = {fill, fill, fill};
//             uint8_t v4[3] = {fill, fill, fill};

//             if (yLow >= 0 && yLow < sharedInfoModel.srcHeight)
//             {
//                 if (xLow >= 0 && xLow < sharedInfoModel.srcWidth)
//                 {
//                     const uint8_t *p = rowBase1 + xLow * 3;
//                     v1[0] = p[0];
//                     v1[1] = p[1];
//                     v1[2] = p[2];
//                 }
//                 if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth)
//                 {
//                     const uint8_t *p = rowBase1 + xHigh * 3;
//                     v2[0] = p[0];
//                     v2[1] = p[1];
//                     v2[2] = p[2];
//                 }
//             }
//             if (yHigh >= 0 && yHigh < sharedInfoModel.srcHeight)
//             {
//                 if (xLow >= 0 && xLow < sharedInfoModel.srcWidth)
//                 {
//                     const uint8_t *p = rowBase2 + xLow * 3;
//                     v3[0] = p[0];
//                     v3[1] = p[1];
//                     v3[2] = p[2];
//                 }
//                 if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth)
//                 {
//                     const uint8_t *p = rowBase2 + xHigh * 3;
//                     v4[0] = p[0];
//                     v4[1] = p[1];
//                     v4[2] = p[2];
//                 }
//             }

//             c0 = w1 * float(v1[0]) + w2 * float(v2[0]) + w3 * float(v3[0]) + w4 * float(v4[0]);
//             c1 = w1 * float(v1[1]) + w2 * float(v2[1]) + w3 * float(v3[1]) + w4 * float(v4[1]);
//             c2 = w1 * float(v1[2]) + w2 * float(v2[2]) + w3 * float(v3[2]) + w4 * float(v4[2]);
//         }
//     }

//     // write hwcImage
//     int hwcIdx = batchIdx * sharedInfoModel.dstHeight * sharedInfoModel.dstWidth * 3 + (y * sharedInfoModel.dstWidth + x)*3;
//     batchHWC[hwcIdx + 0] = c0;
//     batchHWC[hwcIdx + 1] = c1;
//     batchHWC[hwcIdx + 2] = c2;

//     // BGR -> RGB
//     float tmp = batchHWC[hwcIdx + 2];
//     batchHWC[hwcIdx + 2] = batchHWC[hwcIdx + 0];
//     batchHWC[hwcIdx + 0] = tmp;

//     // write planar dst
//     int plane = sharedInfoModel.dstWidth * sharedInfoModel.dstHeight;
//     int dstIdx = batchIdx * 3 * plane + y * sharedInfoModel.dstWidth + x;
//     batchdst[dstIdx + 0 * plane] = c0 / scale;
//     batchdst[dstIdx + 1 * plane] = c1 / scale;
//     batchdst[dstIdx + 2 * plane] = c2 / scale;

// }

// void launchBatchWarpAffineKernel(uint8_t *batchsrc, float *batchdst, float *batchHWC, infoModel *devInfomodel,const float *warpMatrix)
// {
//     dim3 block(16, 16, 16);
//     dim3 grid((devInfomodel->dstWidth + block.x - 1) / block.x, (devInfomodel->dstHeight + block.y - 1) / block.y, (devInfomodel->batchSize + block.z - 1) / block.z);
//     batchWarpAffineKernel<<<grid, block>>>(batchsrc, batchdst, batchHWC, devInfomodel, warpMatrix);
// }