#include "imageProcess.h"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include <numeric>
#include <cuda_runtime_api.h>
#include "logger.h"
#include <algorithm>
#include <cmath>
#include "GPU_utils.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>

/**
 * @brief Preprocess the input image for YOLO model inference
 * @param bbox [in] Bbox object containing configuration and image data
 */
yoloPreprocess::yoloPreprocess()
{
    // Constructor
}
yoloPreprocess::~yoloPreprocess()
{
    // Destructor
}

void yoloPreprocess::run(BBox &bbox, std::vector<float> &inputTensor)
{
    // customLogger::getInstance()->debug("do yoloPreprocess");
    letterBox(bbox, inputTensor);
    //     customLogger::getInstance()->debug("originalWidth: {}, originalHeight: {}", bbox.orinImage.cols, bbox.orinImage.rows);
    //     cv::imwrite("oringImage.jpg", bbox.orinImage);
    cv::imwrite("resizeImage.jpg", bbox.resizeImage);
}

void yoloPreprocess::letterBox(BBox &bbox, std::vector<float> &inputTensor)
{
    customLogger::getInstance()->debug("do letterBox");
    // 取得原始圖像的大小
    // customLogger::getInstance()->debug("originalWidth: {}, originalHeight: {}", bbox.orinImage.cols, bbox.orinImage.rows);
    // 計算縮放比例
    if (bbox.orinImage.empty() || bbox.orinImage.cols <= 0 || bbox.orinImage.rows <= 0)
    {
        customLogger::getInstance()->error("invalid images size image_Size.width : {}, image_Size.height : {}", bbox.orinImage.cols, bbox.orinImage.rows);
        bbox.pad.ratio = 1.0f; // 避免意外的值
        return;
    }
    bbox.pad.ratio = std::min(static_cast<float>(bbox.modelInsize.width) / static_cast<float>(bbox.orinImage.cols), static_cast<float>(bbox.modelInsize.height) / static_cast<float>(bbox.orinImage.rows));
    // customLogger::getInstance()->debug("bbox.modelInsize.width / bbox.orinImage.cols: {}", static_cast<float>(bbox.modelInsize.width) / static_cast<float>(bbox.orinImage.cols));
    // customLogger::getInstance()->debug("bbox.modelInsize.height / bbox.orinImage.rows: {}", static_cast<float>(bbox.modelInsize.height) / static_cast<float>(bbox.orinImage.rows));
    // customLogger::getInstance()->debug("pad.ratio: {}", bbox.pad.ratio);
    // 計算新的圖像大小
    cv::Size newSize;
    newSize = cv::Size(static_cast<int>(std::round(bbox.orinImage.cols * bbox.pad.ratio)), static_cast<int>(std::round(bbox.orinImage.rows * bbox.pad.ratio)));
    // customLogger::getInstance()->debug("new size width: {}", static_cast<int>(std::round(bbox.orinImage.cols * bbox.pad.ratio)));
    // customLogger::getInstance()->debug("new size height: {}", static_cast<int>(std::round(bbox.orinImage.rows * bbox.pad.ratio)));
    // customLogger::getInstance()->debug("newSize: {}", newSize);
    float dw = (static_cast<float>(bbox.modelInsize.width) - (static_cast<float>(newSize.width))) / 2;
    float dh = (static_cast<float>(bbox.modelInsize.height) - (static_cast<float>(newSize.height))) / 2;
    // customLogger::getInstance()->debug("dw: {}, dh: {}", dw, dh);
    cv::Mat resizeImage;
    if (newSize != bbox.orinImage.size())
    {
        // customLogger::getInstance()->debug("image size not equal to new_unpad size do image resize images_size : {}, new size : {}", bbox.orinImage.size(), newSize);
        cv::resize(bbox.orinImage, resizeImage, newSize, cv::INTER_LINEAR);
    }
    else
    {
        // customLogger::getInstance()->debug("image size equal to new_unpad size do image copy images_size : {}, new sie : {}", bbox.orinImage.size(), newSize);
        resizeImage = bbox.orinImage.clone();
    }
    // 計算 padding 的大小
    bbox.pad.left = static_cast<int>(std::floor(dw));
    bbox.pad.right = static_cast<int>(std::ceil(dw));
    bbox.pad.top = static_cast<int>(std::floor(dh));
    bbox.pad.bottom = static_cast<int>(std::ceil(dh));

    // customLogger::getInstance()->debug("pad.left: {}, pad.top: {}, pad.right: {}, pad.bottom: {}", pad.left, pad.top, pad.right, pad.bottom);
    cv::Mat padImage(bbox.modelInsize, CV_8UC3, color);
    // cv::copyMakeBorder(resizeImage, padImage, pad.top, pad.bottom, pad.left, pad.right, cv::BORDER_CONSTANT, color);
    resizeImage.copyTo(padImage(cv::Rect(bbox.pad.left, bbox.pad.top, newSize.width, newSize.height)));
    // cv::imwrite("padImage.jpg", padImage);

    cv::Mat outputwarp;
    cv::Mat warpMatrix;
    float scale;
    int padW;
    int padH;
    scale = bbox.pad.ratio;
    int newWidth = newSize.width;
    int newHeight = newSize.height;
    warpMatrix = (cv::Mat_<float>(2, 3) << scale, 0, dw, 0, scale, dh);
    cv::warpAffine(bbox.orinImage, outputwarp, warpMatrix, bbox.modelInsize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, color);
    // cv::imwrite("outputwarp.jpg", outputwarp);

    // customLogger::getInstance()->debug("padImage size: {}", padImage.size());
    // customLogger::getInstance()->debug("resizeImage size: {}", resizeImage.size());
    bbox.resizeImage = padImage.clone();
    cv::cvtColor(padImage, padImage, cv::COLOR_BGR2RGB);
    padImage.convertTo(padImage, CV_32FC3, 1.0 / 255.0);
    int C = mBindings[0].C;
    int H = mBindings[0].H;
    int W = mBindings[0].W;

    for (int c = 0; c < C; ++c)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                float val = padImage.at<cv::Vec3f>(h, w)[c];
                inputTensor[c * H * W + h * W + w] = val;
            }
        }
    }
}

/**
 * @brief Construct a new yoloPreprocessGPU::yoloPreprocessGPU object
 * @param d_boxes [in] Device pointer for bounding boxes
 * @param bbox [in] Bbox object containing configuration and image data
 */
yoloPreprocessGPU::yoloPreprocessGPU()
{
    // Constructor
    size_t outBytes = static_cast<size_t>(mBindings[0].H * mBindings[0].W * mBindings[0].C * sizeof(float));
    cudaMalloc(&outputImagePtr, outBytes);
    cudaMalloc(&hwcImage, outBytes);
    cudaMalloc(&d_warpMatrix, 6 * sizeof(float));
}

yoloPreprocessGPU::~yoloPreprocessGPU()
{
    // Destructor
    cudaFree(outputImagePtr);
    cudaFree(d_warpMatrix);
    cudaFree(hwcImage);
}
void yoloPreprocessGPU::run(BBox &bbox)
{
    customLogger::getInstance()->debug("do yoloPreProcessGPU");
    // customLogger::getInstance()->debug("originalWidth: {}, originalHeight: {}", bbox.orinImage.cols, bbox.orinImage.rows);
    // cv::imwrite("oringImage.jpg", bbox.orinImage);
    // cv::imwrite("resizeImage.jpg", bbox.resizeImage);
    bbox.gpuInputImage.upload(bbox.orinImage);
    // cv::Mat cpuMat;
    // gpuInputImage.download(cpuMat);
    // cv::namedWindow("GPU Decoded Frame", cv::WINDOW_NORMAL);
    // cv::imshow("GPU Decoded Frame", cpuMat);
    // customLogger::getInstance()->debug("GPUtoCPUWidth: {}, GPUtoCPUHeight: {}", cpuMat.cols, cpuMat.rows);
    // // cv::imwrite("oringImage2.jpg", cpuMat);
    letterBoxGPU(bbox);
}

void yoloPreprocessGPU::letterBoxGPU(BBox &bbox)
{
    customLogger::getInstance()->debug("do letterBoxGPU");
    // 取得原始圖像的大小
    customLogger::getInstance()->debug("cuda process originalWidth: {}, originalHeight: {}", bbox.gpuInputImage.cols, bbox.gpuInputImage.rows);
    customLogger::getInstance()->debug("cuda process modelInsizeWidth: {}, modelInsizeHeight: {}", bbox.modelInsize.width, bbox.modelInsize.height);
    // 計算縮放比例
    if (bbox.gpuInputImage.empty() || bbox.gpuInputImage.cols <= 0 || bbox.gpuInputImage.rows <= 0)
    {
        // customLogger::getInstance()->error("cuda process invalid images size image_Size.width : {}, image_Size.height : {}", inputImage.cols, inputImage.rows);
        bbox.pad.ratio = 1.0f; // 避免意外的值
        return;
    }
    // 計算 padding 的大小
    bbox.pad.ratio = std::min(static_cast<float>(bbox.modelInsize.width) / static_cast<float>(bbox.gpuInputImage.cols), static_cast<float>(bbox.modelInsize.height) / static_cast<float>(bbox.gpuInputImage.rows));
    // 計算新的圖像大小
    cv::Size newSize;
    newSize = cv::Size(static_cast<int>(std::round(bbox.gpuInputImage.cols * bbox.pad.ratio)), static_cast<int>(std::round(bbox.gpuInputImage.rows * bbox.pad.ratio)));
    customLogger::getInstance()->debug("cuda process newSize: {}, {}", newSize.width, newSize.height);
    customLogger::getInstance()->debug("cuda process ratio1: {}", static_cast<float>(bbox.modelInsize.width) / static_cast<float>(bbox.gpuInputImage.cols));
    customLogger::getInstance()->debug("cuda process ratio2: {}", static_cast<float>(bbox.modelInsize.height) / static_cast<float>(bbox.gpuInputImage.rows));
    customLogger::getInstance()->debug("cuda process dw: {}", (static_cast<float>(bbox.modelInsize.width) - (static_cast<float>(newSize.width))) / 2);
    customLogger::getInstance()->debug("cuda process dh: {}", (static_cast<float>(bbox.modelInsize.height) - (static_cast<float>(newSize.height))) / 2);
    customLogger::getInstance()->debug("cuda process pad.ratio: {}", bbox.pad.ratio);
    float dw = (static_cast<float>(bbox.modelInsize.width) - (static_cast<float>(newSize.width))) / 2;
    float dh = (static_cast<float>(bbox.modelInsize.height) - (static_cast<float>(newSize.height))) / 2;
    // cv::cuda::GpuMat resizeImage;
    // if (newSize != bbox.gpuInputImage.size())
    // {
    //     // customLogger::getInstance()->debug("cuda process image size not equal to new_unpad size do image resize images_size : {}, new size : {}", inputImage.size(), pad.pad.newSize);
    //     cv::cuda::resize(bbox.gpuInputImage, resizeImage, bbox.pad.newSize, cv::INTER_LINEAR);
    // }
    // else
    // {
    //     // customLogger::getInstance()->debug("cuda process image size equal to new_unpad size do image copy images_size : {}, new sie : {}", inputImage.size(), pad.pad.newSize);
    //     resizeImage = bbox.gpuInputImage.clone();
    // }
    // customLogger::getInstance()->debug("cuda process pad.ratio: {}", pad.ratio);
    bbox.pad.left = static_cast<int>(std::floor(dw));
    bbox.pad.right = static_cast<int>(std::ceil(dw));
    bbox.pad.top = static_cast<int>(std::floor(dh));
    bbox.pad.bottom = static_cast<int>(std::ceil(dh));
    int srcW = bbox.gpuInputImage.cols;
    int srcH = bbox.gpuInputImage.rows;
    int dstW = bbox.modelInsize.width;
    int dstH = bbox.modelInsize.height;
    float scale = std::min(dstW / (float)srcW, dstH / (float)srcH);
    float tx = (dstW - scale * srcW) * 0.5f;
    float ty = (dstH - scale * srcH) * 0.5f;

    customLogger::getInstance()->debug("inputImage size: {},{}", bbox.gpuInputImage.cols, bbox.gpuInputImage.rows);
    customLogger::getInstance()->debug("modelInsize: {},{}", bbox.modelInsize.width, bbox.modelInsize.height);
    customLogger::getInstance()->debug("cuda process scale: {}, tx: {}, ty: {}", scale, tx, ty);

    warpMatrix = (cv::Mat_<float>(2, 3) << scale, 0.0f, tx,
                  0.0f, scale, ty);
    // customLogger::getInstance()->debug("cuda process warpMatrix: {},{},{},{},{},{}", warpMatrix.at<float>(0, 0), warpMatrix.at<float>(0, 1), warpMatrix.at<float>(0, 2),
    //                                    warpMatrix.at<float>(1, 0), warpMatrix.at<float>(1, 1), warpMatrix.at<float>(1, 2));
    // customLogger::getInstance()->debug("cuda process scale: {}, tx: {}, ty: {}", scale, tx, ty);

    // cv::Mat warpMatrix_inv;
    cv::invertAffineTransform(warpMatrix, warpMatrix_inv);
    // 1. 明確地以 modelInsize 計算記憶體
    const int outW = bbox.modelInsize.width;
    const int outH = bbox.modelInsize.height;
    size_t outBytes = static_cast<size_t>(outW * outH * 3 * sizeof(float));
    // customLogger::getInstance()->debug("cuda process outW: {}, outH: {}, outBytes: {}", outW, outH, outBytes);

    // cudaMalloc(&outputImagePtr, outBytes);
    cudaMemset(outputImagePtr, 0, outBytes);
    assert(outputImagePtr != nullptr);
    // cudaMalloc(&hwcImage, outBytes);
    cudaMemset(hwcImage, 0, outBytes);
    assert(hwcImage != nullptr);
    // 2. 傳入正確大小給 grid
    dim3 block(32, 32);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    // 3. 處理 warpMatrix 複製
    // float *d_warpMatrix = nullptr;
    // cudaMalloc(&d_warpMatrix, 6 * sizeof(float));
    cudaMemcpy(d_warpMatrix, warpMatrix_inv.ptr<float>(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    // customLogger::getInstance()->debug("cuda process warpMatrix: {},{},{},{},{},{}", warpMatrix.at<float>(0, 0), warpMatrix.at<float>(0, 1), warpMatrix.at<float>(0, 2),
    //                                    warpMatrix.at<float>(1, 0), warpMatrix.at<float>(1, 1), warpMatrix.at<float>(1, 2));
    // customLogger::getInstance()->debug("cuda process warpMatrix_inv: {},{},{},{},{},{}", warpMatrix_inv.at<float>(0, 0), warpMatrix_inv.at<float>(0, 1), warpMatrix_inv.at<float>(0, 2),
    //                                    warpMatrix_inv.at<float>(1, 0), warpMatrix_inv.at<float>(1, 1), warpMatrix_inv.at<float>(1, 2));

    // 4. 執行 kernel
    // customLogger::getInstance()->debug("cuda process launchWarpAffineKernel");
    launchWarpAffineKernel(
        bbox.gpuInputImage.ptr<uint8_t>(), outputImagePtr, hwcImage,
        bbox.gpuInputImage.cols, bbox.gpuInputImage.rows,
        static_cast<int>(bbox.gpuInputImage.step),
        outW, outH,
        d_warpMatrix, grid, block);
    // customLogger::getInstance()->debug("inputImage type: {}", inputImage.type());
    // customLogger::getInstance()->debug("static_cast<int>(inputImage.step):{}", static_cast<int>(inputImage.step));
    // customLogger::getInstance()->debug("inputImage.elemSize():{}", inputImage.elemSize());
    // // 5. 直接下載到 CPU
    customLogger::getInstance()->debug("pointer GPU to CPU");
    std::vector<float> cpuBuffer(outW * outH * 3);
    cudaMemcpy(cpuBuffer.data(), hwcImage, outBytes, cudaMemcpyDeviceToHost);

    // // 轉成 Mat
    // customLogger::getInstance()->debug("pointer(float) convter to Mat");
    cv::Mat warpImage(outH, outW, CV_32FC3, cpuBuffer.data());
    cv::Mat warpImageINT;
    warpImage.convertTo(bbox.resizeImage, CV_8UC3, 255.0);
    // cv::namedWindow("outputImage", cv::WINDOW_NORMAL);
    // cv::resizeWindow("outputImage", 640, 640);
    // cv::imshow("outputImage", outputImage);
    // cv::waitKey(0);
    // warpImageINT.copyTo(outputImage);
    cv::imwrite("cuda_process_outwarp.jpg", bbox.resizeImage);

    // 將前處理結果指派到tensorRT input pointer
    // customLogger::getInstance()->debug("pointer convter to tensorRT input");
    cudaMemcpy(mBindings[0].device_ptr, outputImagePtr, outBytes, cudaMemcpyDeviceToDevice);

    // 將 CHW 結果從 GPU 複製到 CPU
    std::vector<float> chwBuffer(outW * outH * 3);
    cudaMemcpy(chwBuffer.data(), mBindings[0].device_ptr, outW * outH * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // // 轉成 HWC 格式
    // std::vector<float> hwcBuffer(outW * outH * 3);
    // for (int y = 0; y < outH; ++y)
    // {
    //     for (int x = 0; x < outW; ++x)
    //     {
    //         int hwc_idx = (y * outW + x) * 3;
    //         int chw_idx = y * outW + x;
    //         hwcBuffer[hwc_idx + 0] = chwBuffer[chw_idx + 0 * outW * outH]; // R
    //         hwcBuffer[hwc_idx + 1] = chwBuffer[chw_idx + 1 * outW * outH]; // G
    //         hwcBuffer[hwc_idx + 2] = chwBuffer[chw_idx + 2 * outW * outH]; // B
    //     }
    // }

    // // 轉成 Mat 顯示
    // cv::Mat hwcwarpImage(outH, outW, CV_32FC3, hwcBuffer.data());
    // cv::Mat hwcwarpImageINT;
    // hwcwarpImage.convertTo(hwcwarpImageINT, CV_8UC3, 255.0);
    // hwcwarpImageINT.copyTo(outputImage);
    // cv::namedWindow("hwcwarpImageINT", cv::WINDOW_NORMAL);
    // cv::resizeWindow("hwcwarpImageINT", 640, 640);
    // cv::imshow("hwcwarpImageINT", hwcwarpImageINT);
    // cv::waitKey(0);
    // // cv::imwrite("cuda_process_chw2hwc_debug.jpg", outputImage);
    // cudaFree(d_warpMatrix);
    // cudaFree(hwcImage);
}

/**
 * @brief Postprocess the output of YOLO model inference
 * @param bbox [in] The bounding boxes
 */
yoloPostprocess::yoloPostprocess()
{
    // Constructor
}

yoloPostprocess::~yoloPostprocess()
{
    // Destructor
    if (outputData)
    {
        free(outputData);
        outputData = nullptr;
    }
}

void yoloPostprocess::run(BBox &bbox)
{
    // 假設 mBindings[1].size, H, C 已正確設置
    // 1. 檢查 outputData 是否需要重新分配
    static size_t lastSize = mBindings[1].size;
    if (outputData == nullptr || mBindings[1].size != lastSize)
    {
        if (outputData)
            free(outputData);
        outputData = (float *)malloc(mBindings[1].size);
    }

    // 2. 檢查 result 是否需要重新分配
    if (result.size() != mBindings[1].H || (result.size() > 0 && result[0].size() != mBindings[1].C))
    {
        result.clear();
        result.resize(mBindings[1].H, std::vector<float>(mBindings[1].C));
    }
    int nClass = mBindings[1].C - 4;

    customLogger::getInstance()->debug("do yoloPostprocess");
    // customLogger::getInstance()->debug("output dims : {}", mBindings[1].dims);
    // customLogger::getInstance()->debug("output size : {}", mBindings[1].size);
    memcpy(outputData, mBindings[1].host_ptr, mBindings[1].size);
    // std::string CPUpointerFile = "yoloPostprocessCPU_pointer.txt";
    // std::ofstream pointerFileCPU(CPUpointerFile);
    // if (!pointerFileCPU.is_open())
    // {
    //     std::cerr << "[ERROR] Failed to open output file: " << CPUpointerFile << std::endl;
    //     return;
    // }
    // for (int i = 0; i < mBindings[1].H; ++i)
    // {
    //     pointerFileCPU << "Row [" << i << "]: ";
    //     for (int j = 0; j < mBindings[1].C; ++j)
    //     {
    //         pointerFileCPU << "result[" << i << "][" << j << "]=" << outputData[i * mBindings[1].C + j] << " ";
    //     }
    //     pointerFileCPU << std::endl;
    // }
    // pointerFileCPU.close();
    // 指標處理成矩陣
    for (int c = 0; c < mBindings[1].C; ++c)
    {
        channelData = &outputData[c * mBindings[1].H];
        for (int h = 0; h < mBindings[1].H; ++h)
        {
            result[h][c] = *channelData;
            channelData++;
        }
    }
    // for (int h = 0; h < mBindings[1].H; ++h)
    // {
    //     for (int c = 0; c < mBindings[1].C; ++c)
    //     {
    //         customLogger::getInstance()->debug("result :{}", result[h][c]);
    //     }
    // }
    // std::string CPUoutFile = "yoloPostprocessCPU_output.txt";
    // std::ofstream outFile(CPUoutFile);
    // if (!outFile.is_open())
    // {
    //     std::cerr << "[ERROR] Failed to open output file: " << CPUoutFile << std::endl;
    //     return;
    // }

    // 將結果轉換為邊界框
    for (int i = 0; i < result.size(); i++)
    {
        auto rowptr = result[i].data();
        auto bboxptr = rowptr;
        auto scoreptr = rowptr + 4;
        auto maxScoreptr = std::max_element(scoreptr, scoreptr + nClass);
        float score = *maxScoreptr;

        Box box;
        box.x = *bboxptr++;
        box.y = *bboxptr++;
        box.w = *bboxptr++;
        box.h = *bboxptr++;
        int label = maxScoreptr - scoreptr;
        box.classId = label;
        box.score = score;
        box.x = (box.x - (box.w / 2));
        box.y = (box.y - (box.h / 2));
        if (score > bbox.cfg.confThreshold)
        {
            boxes.push_back(box);
        }
        // outFile << std::fixed << std::setprecision(6);
        // outFile << "Row [" << i << "]: x=" << box.x << ", y=" << box.y
        //         << ", w=" << box.w << ", h=" << box.h
        //         << ", score=" << box.score
        //         << ", classId=" << box.classId
        //         << std::endl;
    }
    // outFile.close();
    customLogger::getInstance()->debug("do NMS");
    // // 進行非極大值抑制
    NonMaxSuppression(bbox, boxes);
    // customLogger::getInstance()->debug("do decodeBoundingBox");
    // // customLogger::getInstance()->debug("indices size : {}", bbox.indices.size());
    // // customLogger::getInstance()->debug("rect size : {}", bbox.rect.size());
    // // customLogger::getInstance()->debug("classId size : {}", bbox.classId.size());
    // // customLogger::getInstance()->debug("score size : {}", bbox.score.size());

    // // debug反向填充邊界框
    // for (int i = 0; i < bbox.indices.size(); i++)
    // {
    //     // customLogger::getInstance()->debug("indices :{}", bbox.indices.at(i));
    //     // customLogger::getInstance()->debug("rect :{}", boxes[bbox.indices.at(i)].rect);
    //     // customLogger::getInstance()->debug("class :{}", boxes[bbox.indices.at(i)].classId);
    //     // customLogger::getInstance()->debug("score :{}", boxes[bbox.indices.at(i)].score);
    //     float x = boxes[bbox.indices.at(i)].x * bbox.modelInsize.width;
    //     float y = boxes[bbox.indices.at(i)].y * bbox.modelInsize.height;
    //     float w = boxes[bbox.indices.at(i)].w * bbox.modelInsize.width;
    //     float h = boxes[bbox.indices.at(i)].h * bbox.modelInsize.height;
    //     // customLogger::getInstance()->debug("scale model input size x: {}, y: {}, w: {}, h: {}", x, y, w, h);
    //     cv::Rect rect = cv::Rect(x, y, w, h); // pointBox
    //     cv::rectangle(bbox.resizeImage, rect, cv::Scalar(0, 255, 0), 2);
    //     cv::putText(bbox.resizeImage, std::to_string(boxes[bbox.indices.at(i)].classId) + ": " + std::to_string(boxes[bbox.indices.at(i)].score), cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    //     // cv::imwrite("resizeImage_bbox_rect.jpg", bbox.resizeImage);
    // }
    customLogger::getInstance()->debug("do dePadBoxes");
    dePadBoxes(bbox, boxes);

    customLogger::getInstance()->debug("indices size : {}", bbox.indices.size());

    // // 解碼邊界框
    // for (int i = 0; i < bbox.indices.size(); i++)
    // {
    //     customLogger::getInstance()->debug("bbox.rect[{}] : {}", i, bbox.rect.at(i).x);
    //     customLogger::getInstance()->debug("bbox.rect[{}] : {}", i, bbox.rect.at(i).y);
    //     customLogger::getInstance()->debug("bbox.rect[{}] : {}", i, bbox.rect.at(i).width);
    //     customLogger::getInstance()->debug("bbox.rect[{}] : {}", i, bbox.rect.at(i).height);
    //     customLogger::getInstance()->debug("bbox.classId[{}] : {}", i, bbox.classId.at(i));
    //     customLogger::getInstance()->debug("bbox.score[{}] : {}", i, bbox.score.at(i));
    // }
}

float yoloPostprocess::ComputeIoU(const cv::Rect_<float> &box1, const cv::Rect_<float> &box2)
{
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    // customLogger::getInstance()->debug("box1 x: {}, y: {}, w: {}, h: {}", box1.x, box1.y, box1.width, box1.height);
    // customLogger::getInstance()->debug("box2 x: {}, y: {}, w: {}, h: {}", box2.x, box2.y, box2.width, box2.height);
    // customLogger::getInstance()->debug("x1: {}, y1: {}, x2: {}, y2: {}", x1, y1, x2, y2);

    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    // customLogger::getInstance()->debug("intersectionArea: {}", intersectionArea);
    // float box1Area = box1.area();
    // float box2Area = box2.area();

    float box1Area = box1.width * box1.height;
    float box2Area = box2.width * box2.height;
    // customLogger::getInstance()->debug("box1 area: {}, box2 area: {}, intersection area: {}", box1Area, box2Area, intersectionArea);
    float unionArea = box1Area + box2Area - intersectionArea;
    if (unionArea <= 0.0f)
    {                // 防 0 除
        return 0.0f; // 或 return std::numeric_limits<float>::quiet_NaN();
    }
    float iou = intersectionArea / unionArea;
    return iou;
}

void yoloPostprocess::NonMaxSuppression(BBox &bbox, std::vector<Box> &boxes)
{
    customLogger::getInstance()->debug("do NonMaxSuppression");
    std::vector<int> sortIndex(boxes.size());
    std::iota(sortIndex.begin(), sortIndex.end(), 0); // 自動填入 0, 1, ..., boxes.size() - 1
    std::sort(sortIndex.begin(), sortIndex.end(), [&](int a, int b)
              { return boxes.at(a).score > boxes.at(b).score; });
    std::vector<bool> selected(boxes.size(), false);

    // for (int i = 0; i < sortIndex.size(); i++)
    // {
    //     customLogger::getInstance()->debug("boxes[{}].score: {},classId: {}", sortIndex[i], boxes.at(sortIndex[i]).score,boxes.at(sortIndex[i]).classId);
    // }

    for (int i = 0; i < sortIndex.size(); i++)
    {
        int idx_i = sortIndex[i];
        if (selected.at(idx_i))
        {
            continue;
        }

        bbox.indices.emplace_back(idx_i); // 保留此框
        selected.at(idx_i) = true;

        for (int j = i + 1; j < sortIndex.size(); j++)
        {
            int idx_j = sortIndex[j];
            if (selected.at(idx_j))
            {
                continue;
            }
            float iou = ComputeIoU(cv::Rect_<float>(boxes.at(idx_i).x, boxes.at(idx_i).y, boxes.at(idx_i).w, boxes.at(idx_i).h),
                                   cv::Rect_<float>(boxes.at(idx_j).x, boxes.at(idx_j).y, boxes.at(idx_j).w, boxes.at(idx_j).h));
            // float iou = ComputeIoU(boxes.at(idx_i).rect, boxes.at(idx_j).rect);
            // customLogger::getInstance()->debug("IDX_i: {}, rect: {}", idx_i, boxes.at(idx_i).rect);
            // customLogger::getInstance()->debug("IDX_j: {}, rect: {}", idx_j, boxes.at(idx_j).rect);

            // customLogger::getInstance()->debug("iou: {}, bbox.cfg.iouThreshold: {}", iou, bbox.cfg.iouThreshold);
            if (iou > bbox.cfg.iouThreshold)
            {
                selected.at(idx_j) = true;
            }
        }
    }
}

void yoloPostprocess::dePadBoxes(BBox &bbox, std::vector<Box> &boxes)
{
    customLogger::getInstance()->debug("do dePadBoxes");
    for (int i = 0; i < bbox.indices.size(); i++)
    {
        // customLogger::getInstance()->debug("x:{}, y:{}, w:{}, h:{}", boxes[bbox.indices.at(i)].x, boxes[bbox.indices.at(i)].y, boxes[bbox.indices.at(i)].w, boxes[bbox.indices.at(i)].h);
        // 還原邊界框的坐標
        float x = boxes[bbox.indices.at(i)].x * bbox.modelInsize.width;
        float y = boxes[bbox.indices.at(i)].y * bbox.modelInsize.height;
        float w = boxes[bbox.indices.at(i)].w * bbox.modelInsize.width;
        float h = boxes[bbox.indices.at(i)].h * bbox.modelInsize.height;
        customLogger::getInstance()->debug("scale model input size x: {}, y: {}, w: {}, h: {}", x, y, w, h);
        // 反向填充邊界框
        x = std::clamp(x - bbox.pad.left, 0.0f, static_cast<float>(bbox.modelInsize.width));
        y = std::clamp(y - bbox.pad.top, 0.0f, static_cast<float>(bbox.modelInsize.height));
        w = std::clamp(w, 0.0f, static_cast<float>(bbox.modelInsize.width));
        h = std::clamp(h, 0.0f, static_cast<float>(bbox.modelInsize.height));
        // customLogger::getInstance()->debug("unpad  input size x: {}, y: {}, w: {}, h: {}", x, y, w, h);
        // customLogger::getInstance()->debug("bbox pad left:{}, top:{}", bbox.pad.left, bbox.pad.top);
        // // 縮放回原圖片尺寸
        float scaleWidth = static_cast<float>(bbox.orinImage.cols) / static_cast<float>(bbox.modelInsize.width);
        float scaleHeight = static_cast<float>(bbox.orinImage.rows) / static_cast<float>(bbox.modelInsize.height);
        // customLogger::getInstance()->debug("scaleWidth: {}, scaleHeight: {}", scaleWidth, scaleHeight);

        // float scaleX = x * scaleWidth;
        // float scaleY = y * scaleHeight;
        // float scaleW = w * scaleWidth;
        // float scaleH = h * scaleHeight;
        // customLogger::getInstance()->debug("scale original input size x: {}, y: {}, w: {}, h: {}", scaleX, scaleY, scaleW, scaleH);
        // std::cout << std::endl;

        // 將邊界框轉換為 cv::Rect
        // bbox.rect.push_back(cv::Rect(scaleX, scaleY, scaleW, scaleH));
        bbox.rect.push_back(cv::Rect(x, y, w, h));
        bbox.classId.push_back(boxes[bbox.indices.at(i)].classId);
        bbox.score.push_back(boxes[bbox.indices.at(i)].score);
        // customLogger::getInstance()->debug("bbox : {}, {}, {}, {}", x, y, w, h);
    }
}

/**
 * @brief Construct a new yoloPostprocessGPU::yoloPostprocessGPU object
 * @param d_boxes [in] Device pointer for bounding boxes
 * @param bbox [in] Bbox object containing configuration and image data
 */
yoloPostprocessGPU::yoloPostprocessGPU()
{
    // Constructor
    cudaError_t err = cudaMalloc(&d_boxes, sizeof(Box) * mBindings[1].H);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    }
    // else
    // {
    //     std::cout << "[Constructor] d_boxes allocated at: " << d_boxes << std::endl;
    // }
}
yoloPostprocessGPU::~yoloPostprocessGPU()
{
    // Destructor
    cudaFree(d_boxes);
}
void yoloPostprocessGPU::run(BBox &bbox, Pad &pad)
{
    // Implementation
    customLogger::getInstance()->debug("do yoloPostprocessGPU");
    // int nClass = mBindings[1].C - 4;
    // customLogger::getInstance()->debug("mBindings[1].H: {}", mBindings[1].H);
    // customLogger::getInstance()->debug("mBindings[1].C: {}", mBindings[1].C);
    // customLogger::getInstance()->debug("nClass: {}", nClass);
    if (mBindings[1].device_ptr == nullptr)
    {
        std::cerr << "[ERROR] mBindings[1].device_ptr is nullptr!" << std::endl;
    }
    cudaMemset(d_boxes, 0, sizeof(Box) * mBindings[1].H);

    // float *outputDataGPU = (float *)malloc(mBindings[1].H * mBindings[1].C * sizeof(float));
    // cudaMemcpy(outputDataGPU, reinterpret_cast<float *>(mBindings[1].device_ptr), mBindings[1].H * mBindings[1].C * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string GPUpointerFile = "yoloPostprocessGPU_pointer.txt";
    // std::ofstream outFileGPU(GPUpointerFile);
    // if (!outFileGPU.is_open())
    // {
    //     std::cerr << "[ERROR] Failed to open output file: " << GPUpointerFile << std::endl;
    //     return;
    // }
    // for (int i = 0; i < mBindings[1].H; ++i)
    // {
    //     outFileGPU << std::fixed << std::setprecision(6);
    //     outFileGPU << "Row [" << i << "]: ";
    //     for (int c = 0; c < mBindings[1].C; ++c)
    //     {
    //         outFileGPU << "result[" << i << "][" << c << "]=" << outputDataGPU[i * mBindings[1].C + c] << ", ";
    //     }
    //     outFileGPU << std::endl;
    // }
    // outFileGPU.close();
    // free(outputDataGPU);

    launchDecodeBoxesKernel(reinterpret_cast<float *>(mBindings[1].device_ptr), d_boxes, bbox.cfg.confThreshold, mBindings[1].H, mBindings[1].C);

    launchNMSKernel(d_boxes, mBindings[1].H, bbox.cfg.iouThreshold);

    launchUnscale(d_boxes, mBindings[1].H, pad.left, pad.top, bbox.modelInsize.width, bbox.modelInsize.height, bbox.orinImage.cols, bbox.orinImage.rows);

    // debug decodeboxes
    Box *host_boxes = new Box[mBindings[1].H];
    memset(host_boxes, 0, sizeof(Box) * mBindings[1].H);
    cudaMemcpy(host_boxes, d_boxes, sizeof(Box) * mBindings[1].H, cudaMemcpyDeviceToHost);
    // std::string GPUoutFile = "yoloPostprocessGPU_output.txt";
    // std::ofstream outFile(GPUoutFile);
    // if (!outFile.is_open())
    // {
    //     std::cerr << "[ERROR] Failed to open output file: " << GPUoutFile << std::endl;
    //     return;
    // }
    // for (int i = 0; i < mBindings[1].H; ++i)
    // {
    //     outFile << std::fixed << std::setprecision(6);
    //     outFile << "Box[" << i << "]: x=" << host_boxes[i].x
    //             << " y=" << host_boxes[i].y
    //             << " w=" << host_boxes[i].w
    //             << " h=" << host_boxes[i].h
    //             << " score=" << host_boxes[i].score
    //             << " classId=" << host_boxes[i].classId
    //             << " keep=" << host_boxes[i].keep << std::endl;
    // }
    // outFile.close();

    int countKeep = 0;
    for (int i = 0; i < mBindings[1].H; ++i)
    {
        if (host_boxes[i].keep == 0)
            continue;
        else
        {
            BBox box;
            bbox.classId.push_back(host_boxes[i].classId);
            bbox.indices.push_back(i);
            bbox.rect.push_back(cv::Rect(static_cast<int>(host_boxes[i].x), static_cast<int>(host_boxes[i].y), static_cast<int>(host_boxes[i].w), static_cast<int>(host_boxes[i].h)));
            bbox.score.push_back(host_boxes[i].score);
            countKeep++;
        }
    }

    // customLogger::getInstance()->debug("countKeep: {}", countKeep);
    for (int i = 0; i < bbox.indices.size(); i++)
    {
        customLogger::getInstance()->debug("host_boxes[{}].rect: {}", i, bbox.rect[i]);
        customLogger::getInstance()->debug("host_boxes[{}].indices: {}", i, bbox.indices[i]);
        customLogger::getInstance()->debug("host_boxes[{}].score: {}", i, bbox.score[i]);
        customLogger::getInstance()->debug("host_boxes[{}].classId: {}", i, bbox.classId[i]);
    }
    customLogger::getInstance()->debug("bbox.pad.left): {}", bbox.pad.left);
    customLogger::getInstance()->debug("bbox.pad.top): {}", bbox.pad.top);
    customLogger::getInstance()->debug("bbox.pad.right): {}", bbox.pad.right);
    customLogger::getInstance()->debug("bbox.pad.bottom): {}", bbox.pad.bottom);

    free(host_boxes);
}
