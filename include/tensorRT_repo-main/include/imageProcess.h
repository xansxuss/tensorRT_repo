#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H
#pragma once

#define LIBRARY_VERSION_MAJOR 0
#define LIBRARY_VERSION_MINOR 1
#define LIBRARY_VERSION_PATCH 0


#include <stdint.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "struct_type.h"

class yoloPreprocess
{
private:
    float minNumber = 0.01;
    cv::Scalar color = cv::Scalar(114, 114, 114);
    void letterBox(BBox &bbox, std::vector<float> &inputTensor);

public:
    yoloPreprocess();
    ~yoloPreprocess();
    void run(BBox &bbox, std::vector<float> &inputTensor);
};

class yoloPostprocess
{
private:
    void NonMaxSuppression(BBox &bbox, std::vector<Box> &boxes);
    void softNonMaxSuppression(BBox &bbox, std::vector<Box> &boxes);
    float ComputeIoU(const cv::Rect_<float> &box1, const cv::Rect_<float> &box2);
    void DecodeBoundingBox();
    void dePadBoxes(BBox &bbox, std::vector<Box> &boxes);
    float *outputData = nullptr;
    float *channelData = nullptr;
    std::vector<std::vector<float>> result;
    std::vector<Box> boxes;

public:
    yoloPostprocess();
    ~yoloPostprocess();
    void run(BBox &bbox);
};

class yoloPreprocessGPU
{
private:
    cv::Scalar color = cv::Scalar(114, 114, 114);
    float *outputImagePtr = nullptr;
    float *d_warpMatrix = nullptr;
    float *hwcImage = nullptr;

    // cudaStream_t stream;
    void letterBoxGPU(BBox &bbox);

public:
    cv::Mat warpMatrix_inv;
    cv::Mat warpMatrix;
    yoloPreprocessGPU();
    ~yoloPreprocessGPU();
    void run(BBox &bbox);
};

class yoloPostprocessGPU
{
private:
    Box *d_boxes = nullptr;
    Box *host_boxes;
    float *d_warpMatrix = nullptr;
    //     void NonMaxSuppressionGPU();
    //     void softNonMaxSuppressionGPU();
    //     float ComputeIoUGPU();
    //     void DecodeBoundingBoxGPU();
    //     void dePadBoxesGPU();
public:
    yoloPostprocessGPU();
    ~yoloPostprocessGPU();
    void run(BBox &bbox);
};

#endif