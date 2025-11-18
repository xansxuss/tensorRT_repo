#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
# include "struct_type.h"

#pragma once

// 前處理：
__global__ void warpAffineKernel(uint8_t *src, float *dst,float* hwcImage, int srcWidth, int srcHeight,int srcPitch, int dstWidth, int dstHeight, const float *warpMatrix,float constValueSt,uint8_t constValue[3]);
void launchWarpAffineKernel(uint8_t *src, float *dst,float* hwcImage, int srcWidth, int srcHeight,int srcPitch, int dstWidth, int dstHeight, const float *warpMatrix, dim3 grid, dim3 block);

// 後處理：
__global__ void transposeCHtoHC(const float* src, float* dst, int C, int H);
void launchTransposeCHtoHC(float* src, float* dst, int C, int H, dim3 grid, dim3 block);

__global__ void decodeKernel(float *predict, int numBoxes, int numClass, float confidenceThreshold, int *selectIndices, int *outCount, Box *outBoxes);
void launchDecodeKernel(float *d_predict,int numBoxes,int numClass,float confidenceThreshold,int *d_selectIndices,int *d_outCount,Box *d_outBoxes);

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom);
__global__ void nmsKernel(Box* boxes, int numBoxes, float iouThreshold, int* selectedCount, int* selectedIndices);
void launchNMSKernel(Box* d_boxes, int numBoxes, float iouThreshold, int* d_selectedCount, int* d_selectedIndices);
#endif