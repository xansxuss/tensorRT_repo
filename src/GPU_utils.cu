#include "GPU_utils.cuh"
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

// 前處理：
// __constant__ uint8_t d_constValue[3];
// GPU 核心函數
__global__ void warpAffineKernel(
    uint8_t *src, float *dst, float *hwcImage,
    int srcWidth, int srcHeight,
    int srcPitch,
    int dstWidth, int dstHeight,
    const float *warpMatrix)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    uint8_t d_constValue[3] = {114, 114, 114};

    if (x >= dstWidth || y >= dstHeight)
    {
        return;
    }

    float mX1 = warpMatrix[0];
    float mY1 = warpMatrix[1];
    float mZ1 = warpMatrix[2];
    float mX2 = warpMatrix[3];
    float mY2 = warpMatrix[4];
    float mZ2 = warpMatrix[5];

    float srcX = mX1 * x + mY1 * y + mZ1;
    float srcY = mX2 * x + mY2 * y + mZ2;

    float c0 = 0, c1 = 0, c2 = 0;

    if (srcX <= -1 || srcX >= srcWidth -1  || srcY <= -1 || srcY >= srcHeight -1 )
    {
        // 這裡直接填預設顏色
        c0 = 114.0f;
        c1 = 114.0f;
        c2 = 114.0f;
    }
    else
    {
        int xLow = floorf(srcX);
        int yLow = floorf(srcY);
        int xHigh = xLow + 1;
        int yHigh = yLow + 1;

        float lx = srcX - xLow;
        float ly = srcY - yLow;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        float w1 = hx * hy;
        float w2 = lx * hy;
        float w3 = hx * ly;
        float w4 = lx * ly;

        // printf("xLow : %d, yLow : %d, XHigh : %d, YHigh : %d \n", xLow, yLow, xHigh, yHigh);
        // printf("lx : %.2f, ly : %.2f, hx : %.2f, hy : %.2f \n", lx, ly, hx, hy);

        uint8_t *v1 = d_constValue;
        uint8_t *v2 = d_constValue;
        uint8_t *v3 = d_constValue;
        uint8_t *v4 = d_constValue;

        if (yLow >= 0 && yLow < srcHeight)
        {
            if (xLow >= 0 && xLow < srcWidth)
                v1 = src + yLow * srcPitch + xLow * 3;
            if (xHigh >= 0 && xHigh < srcWidth)
                v2 = src + yLow * srcPitch + xHigh * 3;
        }
        if (yHigh >= 0 && yHigh < srcHeight)
        {
            if (xLow >= 0 && xLow < srcWidth)
                v3 = src + yHigh * srcPitch + xLow * 3;
            if (xHigh >= 0 && xHigh < srcWidth)
                v4 = src + yHigh * srcPitch + xHigh * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }
    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // debug show
    int dstIdx = (y * dstWidth + x) * 3;
    hwcImage[dstIdx + 0] = c0;
    hwcImage[dstIdx + 1] = c1;
    hwcImage[dstIdx + 2] = c2;

    // BGR to RGB
    float t = c2;
    c2 = c0;
    c0 = t;

    int idx = y * dstWidth + x;
    dst[idx + 0 * dstWidth * dstHeight] = c0; // R
    dst[idx + 1 * dstWidth * dstHeight] = c1; // G
    dst[idx + 2 * dstWidth * dstHeight] = c2; // B
}

// 2. Host function：準備常數並呼叫 kernel
void launchWarpAffineKernel(
    uint8_t *src, float *dst, float *hwcImage,
    int srcWidth, int srcHeight,
    int srcPitch,
    int dstWidth, int dstHeight,
    const float *warpMatrix,
    dim3 grid, dim3 block)
{
    // std::cout << "Matrix: " << warpMatrix[0] << ", " << warpMatrix[1] << ", " << warpMatrix[2] << ", "
    //           << warpMatrix[3] << ", " << warpMatrix[4] << ", " << warpMatrix[5] << std::endl;

    // 呼叫 kernel
    warpAffineKernel<<<grid, block>>>(
        src, dst, hwcImage, srcWidth, srcHeight, srcPitch, dstWidth, dstHeight, warpMatrix);

    // 同步與錯誤檢查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize(); // 確保完成
    // std::cout << "warpAffineKernel execution complete." << std::endl;
}

// 後處理:
// 1.decode kernel

__global__ void decodeBoxesKernel(float *outputptr, Box *boxes, float scoreThreshold, int numBoxes, int numAttrs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes)
    {
        return;
    }
    int numClass = numAttrs - 4; // 4 個屬性是 x, y, w, h
    // 每個 box 的起始位置
    // output layout: [84][8400] => output[attr * num_boxes + box_index]
    float x = outputptr[0 * numBoxes + idx];
    float y = outputptr[1 * numBoxes + idx];
    float w = outputptr[2 * numBoxes + idx];
    float h = outputptr[3 * numBoxes + idx];
    // float obj_score = outputptr[4 * numBoxes + idx];

    int classId = -1;
    float maxScore = -1.0f;
    for (int i = 4; i < numClass; i++)
    {
        float score = outputptr[i * numBoxes + idx];
        if (score > maxScore)
        {
            maxScore = score;
            classId = i - 4; // 因為前面有 4 個屬性
        }
    }
    // 將結果寫入 boxes
    Box box;
    box.x = (x - (w / 2));
    box.y = (y - (h / 2));
    box.w = w;
    box.h = h;
    box.classId = classId;
    box.score = maxScore;
    if (maxScore < scoreThreshold)
    {
        box.keep = 0;
    }
    else
    {
        box.keep = 1;
    }
    boxes[idx] = box;
}

struct BoxCompare
{
    __device__ bool operator()(const Box &a, const Box &b) const
    {
        return a.score > b.score;
    }
};

void sortBoxesByScore(Box *d_boxes, int numBoxes)
{
    thrust::device_ptr<Box> dev_ptr = thrust::device_pointer_cast(d_boxes);

    thrust::sort(thrust::device, d_boxes, d_boxes + numBoxes, BoxCompare());
}

void launchDecodeBoxesKernel(float *outputptr, Box *boxes, float scoreThreshold, int numBoxes, int numAttrs)
{
    dim3 block(256);
    dim3 grid((numBoxes + block.x - 1) / block.x);
    decodeBoxesKernel<<<grid, block>>>(outputptr, boxes, scoreThreshold, numBoxes, numAttrs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize(); // 確保完成
    // sortBoxesByScore(boxes, numBoxes);
}
// 2.NMS Unit
__device__ float IOU(Box &a, Box &b)
{
    float x1 = max(a.x, b.x);
    float y1 = max(a.y, b.y);
    float x2 = min(a.x + a.w, b.x + b.w);
    float y2 = min(a.y + a.h, b.y + b.h);

    float interWidth = max(0.0f, x2 - x1);
    float interHeight = max(0.0f, y2 - y1);
    float interArea = interWidth * interHeight;

    float unionArea = a.w * a.h + b.w * b.h - interArea;
    return interArea / unionArea;
}
__global__ void nmsKernel(Box *d_boxes, int numBoxes, float iou_threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes || d_boxes[idx].keep == 0)
    {
        return;
    }

    for (int j = 0; j < idx; ++j)
    {
        if (d_boxes[j].keep && IOU(d_boxes[idx], d_boxes[j]) > iou_threshold)
        {
            d_boxes[idx].keep = 0;
            return; // 不需要再比下去了
        }
    }
    // 若通過所有 IoU 檢查，則保留
    d_boxes[idx].keep = 1;
}

void launchNMSKernel(Box *d_boxes, int numBoxes, float iou_threshold)
{
    int threads = 256;
    int blocks = (numBoxes + threads - 1) / threads;
    nmsKernel<<<blocks, threads>>>(d_boxes, numBoxes, iou_threshold);
    cudaDeviceSynchronize();
}

__global__ void unscaleKernel(Box *d_boxes, int numBoxes, float left, float top, float modelWidth, float modelHeight, float width, float height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes || d_boxes[idx].keep == 0)
    {
        return;
    }
    d_boxes[idx].x = (d_boxes[idx].x * modelWidth) - left;
    d_boxes[idx].y = (d_boxes[idx].y * modelHeight) - top;
    d_boxes[idx].w = d_boxes[idx].w * modelWidth;
    d_boxes[idx].h = d_boxes[idx].h * modelHeight;
}

void launchUnscale(Box *d_boxes, int numBoxes, float left, float top, float modelWidth, float modelHeight, float width, float height)
{
    int threads = 265;
    int blocks = (numBoxes + threads - 1) / threads;
    unscaleKernel<<<blocks, threads>>>(d_boxes, numBoxes, left, top, modelWidth, modelHeight, width, height);
    // printf("modelWidth: %f, modelHeight: %f\n", modelWidth, modelHeight);
    // printf("width: %f, height: %f\n", width, height);
    cudaDeviceSynchronize();
}