#include "GPU_utils.cuh"
#include <stdint.h>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include "struct_type.h"




// 前處理：
// 1. 使用 __constant__ 宣告 GPU 上的常數
__constant__ float d_constValueSt;
__constant__ uint8_t d_constValue[3];

// 2. GPU 核心函數
__global__ void warpAffineKernel(
    uint8_t* src, float* dst,float* hwcImage,
    int srcWidth, int srcHeight,
    int srcPitch,
    int dstWidth, int dstHeight,
    const float* warpMatrix)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dstWidth || y >= dstHeight)
        {return;}

    float mX1 = warpMatrix[0];
    float mY1 = warpMatrix[1];
    float mZ1 = warpMatrix[2];
    float mX2 = warpMatrix[3];
    float mY2 = warpMatrix[4];
    float mZ2 = warpMatrix[5];

    float srcX = mX1 * x + mY1 * y + mZ1;
    float srcY = mX2 * x + mY2 * y + mZ2 ;

    float c0 = 0, c1 = 0, c2 = 0;

    if (srcX <= -1 || srcX >= srcWidth || srcY <= -1 || srcY >= srcHeight)
    {
    // 這裡直接填預設顏色
    c0 = 114.0f / 255.0f;
    c1 = 114.0f / 255.0f;
    c2 = 114.0f / 255.0f;
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

        uint8_t* v1 = d_constValue;
        uint8_t* v2 = d_constValue;
        uint8_t* v3 = d_constValue;
        uint8_t* v4 = d_constValue;

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

    //BGR to RGB
    float t =c2;
    c2 = c0;
    c0 = t;

    int idx = y * dstWidth + x;
    dst[idx + 0 * dstWidth * dstHeight] = c0; // R
    dst[idx + 1 * dstWidth * dstHeight] = c1; // G
    dst[idx + 2 * dstWidth * dstHeight] = c2; // B
}

// 3. Host function：準備常數並呼叫 kernel
void launchWarpAffineKernel(
    uint8_t* src, float* dst,float* hwcImage,
    int srcWidth, int srcHeight,
    int srcPitch,
    int dstWidth, int dstHeight,
    const float* warpMatrix,
    dim3 grid, dim3 block)
{
    // // 安全傳遞常數值到 GPU
    // float h_constValueSt = 114.0f;
    // uint8_t h_constValue[3] = {114, 114, 114};
    // cudaMemcpyToSymbol(d_constValueSt, &h_constValueSt, sizeof(float));
    // cudaMemcpyToSymbol(d_constValue, h_constValue, sizeof(uint8_t) * 3);

    // std::cout << "Matrix: " << warpMatrix[0] << ", " << warpMatrix[1] << ", " << warpMatrix[2] << ", "
    //           << warpMatrix[3] << ", " << warpMatrix[4] << ", " << warpMatrix[5] << std::endl;

    // 呼叫 kernel
    warpAffineKernel<<<grid, block>>>(
        src, dst, hwcImage,srcWidth, srcHeight,srcPitch, dstWidth ,dstHeight, warpMatrix);

    // 同步與錯誤檢查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            }
    cudaDeviceSynchronize();  // 確保完成
    // std::cout << "warpAffineKernel execution complete." << std::endl;
}


//後處理:
//1.轉置 kernel
__global__ void transposeCHtoHC(float* src, float* dst, int C, int H){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= C || h >= H) {return;}
    dst[h*C+c] = src[c*H+h];
    // printf("Transposing C: %d, H: %d to dst[%d][%d]\n", c, h, h, c);
}

// 2. Host function：呼叫轉置 kernel
void launchTransposeCHtoHC(float* src, float* dst, int C, int H, dim3 grid, dim3 block) {
    transposeCHtoHC<<<grid, block>>>(src, dst, C, H);
    
    // 同步與錯誤檢查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Transpose kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();  // 確保完成
    std::cout << "transposeCHtoHC execution complete." << std::endl;

    int numBoxes = 8400; // 假設有 8400 個框
    int numClass = 80;  // 假設有 80 個類別
    int* d_selectIndices;
    int* d_outCount;
    Box* d_outBoxes;
    int maxOutput = numBoxes;

    // 分配輸出緩衝區
    cudaMalloc((void**)&d_selectIndices, maxOutput * sizeof(int));
    cudaMalloc((void**)&d_outCount, sizeof(int));
    cudaMalloc((void**)&d_outBoxes, maxOutput * sizeof(Box));
    cudaMemset(d_outCount, 0, sizeof(int));
    launchDecodeKernel(dst, numBoxes, numClass, 0.5f, d_selectIndices, d_outCount, d_outBoxes);
    std::cout<<"launchDecodeKernel completed."<<std::endl;
    // 檢查輸出
    int h_outCount = 0;
    cudaMemcpy(&h_outCount, d_outCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Number of detected boxes: " << h_outCount << std::endl;
    if (h_outCount > 0) {
        std::vector<int> h_selectIndices(h_outCount);
        std::vector<Box> h_outBoxes(h_outCount);
        cudaMemcpy(h_selectIndices.data(), d_selectIndices, h_outCount * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_outBoxes.data(), d_outBoxes, h_outCount * sizeof(Box), cudaMemcpyDeviceToHost);

        for (int i = 0; i < h_outCount; ++i) {
            std::cout << "Box " << i << ": [x=" << h_outBoxes[i].x
                      << ", y=" << h_outBoxes[i].y
                      << ", w=" << h_outBoxes[i].w
                      << ", h=" << h_outBoxes[i].h
                      << ", classId=" << h_outBoxes[i].classId
                      << ", score=" << h_outBoxes[i].score
                      << "] at index " << h_selectIndices[i] << std::endl;
        }
    } else {
        std::cout << "No boxes detected." << std::endl;
    }

    // 呼叫 NMS kernel
    float iouThreshold = 0.5f; // IOU 閾值
    int* d_selectedCount;
    int* d_selectedIndices;
    cudaMalloc((void**)&d_selectedCount, sizeof(int));
    cudaMalloc((void**)&d_selectedIndices, maxOutput * sizeof(int));
    cudaMemset(d_selectedCount, 0, sizeof(int));
    launchNMSKernel(d_outBoxes, h_outCount, iouThreshold, d_selectedCount, d_selectedIndices);
    std::cout << "launchNMSKernel completed." << std::endl;


    // 釋放資源
    cudaFree(d_selectIndices);
    cudaFree(d_outCount);
    cudaFree(d_outBoxes);
    cudaFree(d_selectedCount);
    cudaFree(d_selectedIndices);

}

//篩選高分數框
__global__ void decodeKernel(float *predict, int numBoxes, int numClass, float confidenceThreshold, int *selectIndices, int *outCount, Box *outBoxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;

    float* rowptr = predict + idx * (4 + numClass);
    float* bboxptr = rowptr;
    float* scoreptr = rowptr + 4;

    // 找最大分數與類別
    float maxScore = scoreptr[0];
    int label = 0;
    for (int c = 1; c < numClass; ++c) {
        if (scoreptr[c] > maxScore) {
            maxScore = scoreptr[c];
            label = c;
        }
    }

    if (maxScore > confidenceThreshold) {
        // 原子操作取得寫入位置
        int outIdx = atomicAdd(outCount, 1);
        

        // 取出 box 資訊
        float x = bboxptr[0];
        float y = bboxptr[1];
        float w = bboxptr[2];
        float h = bboxptr[3];
        

        // 轉換座標
        Box box;
        box.x = x - w * 0.5f;
        box.y = y - h * 0.5f;
        box.w = w;
        box.h = h;
        box.classId = label;
        box.score = maxScore;
        outBoxes[outIdx] = box;
        selectIndices[outIdx] = idx;
    }
}

// Host function：呼叫 decode kernel
void launchDecodeKernel(float *d_predict,int numBoxes,int numClass,float confidenceThreshold,int *d_selectIndices,int *d_outCount,Box *d_outBoxes)
{
    // 設定 kernel grid/block
    int grid = 256;
    int block = (numBoxes + grid - 1) / grid; // 確保每個 block 處理足夠的 boxes
    // cuda timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    decodeKernel<<<grid, block>>>(d_predict, numBoxes, numClass, confidenceThreshold, d_selectIndices, d_outCount, d_outBoxes);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

        // 錯誤檢查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "decodeKernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "decodeKernel execution time: " << elapsedTime << " ms" << std::endl;

    // // 檢查輸出
    // int h_outCount = 0;
    // cudaMemcpy(&h_outCount, d_outCount, sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Number of detected boxes: " << h_outCount << std::endl;
    // if (h_outCount > 0) {
    //     std::vector<int> h_selectIndices(h_outCount);
    //     std::vector<Box> h_outBoxes(h_outCount);
    //     cudaMemcpy(h_selectIndices.data(), d_selectIndices, h_outCount * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_outBoxes.data(), d_outBoxes, h_outCount * sizeof(Box), cudaMemcpyDeviceToHost);

    //     for (int i = 0; i < h_outCount; ++i) {
    //         std::cout << "Box " << i << ": [x=" << h_outBoxes[i].x
    //                   << ", y=" << h_outBoxes[i].y
    //                   << ", w=" << h_outBoxes[i].w
    //                   << ", h=" << h_outBoxes[i].h
    //                   << ", classId=" << h_outBoxes[i].classId
    //                   << ", score=" << h_outBoxes[i].score
    //                   << "] at index " << h_selectIndices[i] << std::endl;
    //     }
    // } else {
    //     std::cout << "No boxes detected." << std::endl;
    // }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom) {
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void nmsKernel(Box* boxes, int numBoxes, float iouThreshold, int* selectedCount, int* selectedIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;

    Box boxA = boxes[idx];
    if (boxA.score < 0.5f) return; // 只處理高分數框

    for (int j = idx + 1; j < numBoxes; ++j) {
        Box boxB = boxes[j];
        if (boxB.score < 0.5f) continue; // 只處理高分數框

        float iou = box_iou(boxA.x, boxA.y, boxA.x + boxA.w, boxA.y + boxA.h,
                            boxB.x, boxB.y, boxB.x + boxB.w, boxB.y + boxB.h);
        if (iou > iouThreshold) {
            // 如果 IOU 超過閾值，則將 boxB 的分數設為負值，表示被過濾掉
            boxes[j].score = -1.0f;
        }
    }
    // 將未被過濾的框加入選擇列表
    if (boxA.score >= 0.5f) {
        int outIdx = atomicAdd(selectedCount, 1);
        selectedIndices[outIdx] = idx;
    }
}

void launchNMSKernel(Box* d_boxes, int numBoxes, float iouThreshold, int* d_selectedCount, int* d_selectedIndices) {
    int blockSize = 256;
    int numBlocks = (numBoxes + blockSize - 1) / blockSize;

    nmsKernel<<<numBlocks, blockSize>>>(d_boxes, numBoxes, iouThreshold, d_selectedCount, d_selectedIndices);

    // 同步與錯誤檢查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "NMS kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();  // 確保完成
    std::cout << "NMS kernel launched with " << numBlocks << " blocks and " << blockSize << " threads per block." << std::endl;
    // 檢查選擇的框數量
    int h_selectedCount = 0;
    cudaMemcpy(&h_selectedCount, d_selectedCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Number of selected boxes after NMS: " << h_selectedCount << std::endl;
    std::cout << "NMS kernel execution complete." << std::endl;
}