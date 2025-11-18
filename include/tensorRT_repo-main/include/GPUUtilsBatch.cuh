#ifndef GPUUTILSBATCH_CUH
#define GPUUTILSBATCH_CUH
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include "struct_type.h"

struct __align__(16) infoModel
{
    int srcWidth;
    int srcHeight;
    int srcChannel;
    int srcPitch;
    int dstWidth;
    int dstHeight;
    int dstChannel;
    int batchSize;
};

__device__ void
batchWarpAffineKernel(uint8_t *src, float *dst, infoModel &devInfomodel,const float* warpMatrix);

void luanchbatchWarpAffineKernel(uint8_t *src, float *dst, infoModel &devInfomodel);

#endif