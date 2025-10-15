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
__global__ void decodeBoxesKernel(float* outputptr,Box* boxes,float scoreThreshold,int numBoxes,int numAttrs);
void launchDecodeBoxesKernel(float* outputptr, Box* boxes,float scoreThreshold, int numBoxes, int numAttrs);

__global__ void nmsKernel(Box* d_boxes,int numBoxes,float iou_threshold);
void launchNMSKernel(Box* d_boxes,int numBoxes,float iou_threshold);

__global__ void InverseKernel(Box *d_boxes, int numBoxes, const float *warpMatrix,const float scale,float modelWidth, float modelHeight, float width, float height);
void launchInverse(Box *d_boxes, int numBoxes, const float *warpMatrix,const float scale,float modelWidth, float modelHeight, float width, float height);
#endif