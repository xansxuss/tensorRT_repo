#include "your_header_path/gpuutilsbatch.cuh"
#include <math.h>

__global__ void
batchWarpAffineKernel(const uint8_t *batchsrc, float *batchdst, float *batchHWC, const infoModel devInfomodel, const float* warpMatrix)
{
    int batchIdx = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (batchIdx >= devInfomodel.batchSize) return;
    if (x >= devInfomodel.dstWidth || y >= devInfomodel.dstHeight) return;

    const uint8_t fill = 114;

    __shared__ float sharedWarpMatrix[6];
    __shared__ infoModel sharedInfoModel;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // 複製該 batch 的 warp matrix 與 infoModel 到 shared
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            sharedWarpMatrix[i] = warpMatrix[batchIdx * 6 + i];
        }
        sharedInfoModel = devInfomodel;
    }
    __syncthreads();

    // 計算 source 座標
    float srcX = sharedWarpMatrix[0] * x + sharedWarpMatrix[1] * y + sharedWarpMatrix[2];
    float srcY = sharedWarpMatrix[3] * x + sharedWarpMatrix[4] * y + sharedWarpMatrix[5];

    float c0 = 0.f, c1 = 0.f, c2 = 0.f;

    // 修正邊界判斷（注意 -1.0f）
    if (srcX <= -1.0f || srcY <= -1.0f || srcX >= float(sharedInfoModel.srcWidth) - 1.0f || srcY >= float(sharedInfoModel.srcHeight) - 1.0f) {
        c0 = c1 = c2 = float(fill);
    } else {
        int xLow = (int)floorf(srcX);
        int yLow = (int)floorf(srcY);
        int xHigh = xLow + 1;
        int yHigh = yLow + 1;

        float lx = srcX - float(xLow);
        float ly = srcY - float(yLow);
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;
        float w1 = hx * hy;
        float w2 = lx * hy;
        float w3 = hx * ly;
        float w4 = lx * ly;

        bool pitchDiv3 = (sharedInfoModel.srcPitch % 3 == 0);
        if (pitchDiv3) {
            // safe-ish cast when pitch is multiple of 3
            const uchar3* src3 = reinterpret_cast<const uchar3*>(batchsrc);
            int srcRowPixels = sharedInfoModel.srcPitch / 3; // pixels per row
            size_t imageStride = (size_t)sharedInfoModel.srcHeight * (size_t)srcRowPixels;
            size_t base = (size_t)batchIdx * imageStride;

            uchar3 p1 = make_uchar3(fill, fill, fill);
            uchar3 p2 = p1, p3 = p1, p4 = p1;

            if (yLow >= 0 && yLow < sharedInfoModel.srcHeight) {
                if (xLow >= 0 && xLow < sharedInfoModel.srcWidth) {
                    p1 = src3[base + (size_t)yLow * srcRowPixels + xLow];
                }
                if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth) {
                    p2 = src3[base + (size_t)yLow * srcRowPixels + xHigh];
                }
            }
            if (yHigh >= 0 && yHigh < sharedInfoModel.srcHeight) {
                if (xLow >= 0 && xLow < sharedInfoModel.srcWidth) {
                    p3 = src3[base + (size_t)yHigh * srcRowPixels + xLow];
                }
                if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth) {
                    p4 = src3[base + (size_t)yHigh * srcRowPixels + xHigh];
                }
            }

            // p.* 順序取決於來源是 BGR 還是 RGB；假設來源是 BGR (OpenCV)
            c0 = w1 * float(p1.x) + w2 * float(p2.x) + w3 * float(p3.x) + w4 * float(p4.x); // B
            c1 = w1 * float(p1.y) + w2 * float(p2.y) + w3 * float(p3.y) + w4 * float(p4.y); // G
            c2 = w1 * float(p1.z) + w2 * float(p2.z) + w3 * float(p3.z) + w4 * float(p4.z); // R
        } else {
            // byte-wise safe access
            size_t imageRowBytes = (size_t)sharedInfoModel.srcPitch;
            size_t imageBase = (size_t)batchIdx * (size_t)sharedInfoModel.srcHeight * imageRowBytes;

            uint8_t v1[3] = {fill, fill, fill};
            uint8_t v2[3] = {fill, fill, fill};
            uint8_t v3[3] = {fill, fill, fill};
            uint8_t v4[3] = {fill, fill, fill};

            if (yLow >= 0 && yLow < sharedInfoModel.srcHeight) {
                if (xLow >= 0 && xLow < sharedInfoModel.srcWidth) {
                    const uint8_t *p = batchsrc + imageBase + (size_t)yLow * imageRowBytes + (size_t)xLow * 3;
                    v1[0] = p[0]; v1[1] = p[1]; v1[2] = p[2];
                }
                if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth) {
                    const uint8_t *p = batchsrc + imageBase + (size_t)yLow * imageRowBytes + (size_t)xHigh * 3;
                    v2[0] = p[0]; v2[1] = p[1]; v2[2] = p[2];
                }
            }
            if (yHigh >= 0 && yHigh < sharedInfoModel.srcHeight) {
                if (xLow >= 0 && xLow < sharedInfoModel.srcWidth) {
                    const uint8_t *p = batchsrc + imageBase + (size_t)yHigh * imageRowBytes + (size_t)xLow * 3;
                    v3[0] = p[0]; v3[1] = p[1]; v3[2] = p[2];
                }
                if (xHigh >= 0 && xHigh < sharedInfoModel.srcWidth) {
                    const uint8_t *p = batchsrc + imageBase + (size_t)yHigh * imageRowBytes + (size_t)xHigh * 3;
                    v4[0] = p[0]; v4[1] = p[1]; v4[2] = p[2];
                }
            }

            c0 = w1 * float(v1[0]) + w2 * float(v2[0]) + w3 * float(v3[0]) + w4 * float(v4[0]);
            c1 = w1 * float(v1[1]) + w2 * float(v2[1]) + w3 * float(v3[1]) + w4 * float(v4[1]);
            c2 = w1 * float(v1[2]) + w2 * float(v2[2]) + w3 * float(v3[2]) + w4 * float(v4[2]);
        }
    }

    // write HWC: 我們直接寫成 RGB（避免後面再 swap）
    int hwcIdx = batchIdx * sharedInfoModel.dstHeight * sharedInfoModel.dstWidth * 3 + (y * sharedInfoModel.dstWidth + x) * 3;
    // 假設來源是 BGR，現在要輸出 RGB => 把 c2,c1,c0 寫成 R,G,B
    batchHWC[hwcIdx + 0] = c2; // R
    batchHWC[hwcIdx + 1] = c1; // G
    batchHWC[hwcIdx + 2] = c0; // B

    // write planar normalized dst (channel order R,G,B)
    int plane = sharedInfoModel.dstWidth * sharedInfoModel.dstHeight;
    size_t basePlane = (size_t)batchIdx * 3 * (size_t)plane;
    const float scale = 1.0f / 255.0f;
    batchdst[basePlane + 0 * plane + y * sharedInfoModel.dstWidth + x] = c2 * scale; // R
    batchdst[basePlane + 1 * plane + y * sharedInfoModel.dstWidth + x] = c1 * scale; // G
    batchdst[basePlane + 2 * plane + y * sharedInfoModel.dstWidth + x] = c0 * scale; // B
}

void launchBatchWarpAffineKernel(const uint8_t *batchsrc, float *batchdst, float *batchHWC, const infoModel &devInfomodel, const float *warpMatrix)
{
    dim3 block(16, 16);
    dim3 grid( (devInfomodel.dstWidth + block.x - 1) / block.x,
               (devInfomodel.dstHeight + block.y - 1) / block.y,
               devInfomodel.batchSize ); // 加上 z dimension

    // optional: 檢查 grid.z 是否超過 device 最大值（deviceProp.maxGridSize[2]）
    batchWarpAffineKernel<<<grid, block>>>(batchsrc, batchdst, batchHWC, devInfomodel, warpMatrix);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // host side 錯誤處理（只示範）
        printf("launch kernel error: %s\n", cudaGetErrorString(err));
    }
}
