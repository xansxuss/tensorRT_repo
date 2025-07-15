#ifndef STRUCT_TYPE_H
#define STRUCT_TYPE_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

struct Binding
{
    void *device_ptr = nullptr;
    void *host_ptr = nullptr;
    size_t size = 0;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    std::string name;
    int N = 0;
    int C = 0;
    int H = 0;
    int W = 0;
    bool is_input = 0;
};
extern std::vector<Binding> mBindings;

struct yolocfg
{
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    float iouThreshold = 0.5;
    float softNMSSigma = 0.5;
    bool useSoftNMS = true;
};
struct configStruct
{
    std::string enginePath = "./model/yolov8n.engine";
    std::string imagePath = "./media/input/000000005060.jpg";
    std::string savePath = "./media/output";
    std::unordered_map<int,std::string> classNames;
    std::string outputFolder = "";
};
extern configStruct configstruct;

struct Pad
{
    float top = 0.0f;
    float bottom = 0.0f;
    float left = 0.0f;
    float right = 0.0f;
    float ratio = 0.0f;
    float dw = 0.0f;
    float dh = 0.0f;
};

struct Box
{
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float score = 0.0f;
    int classId = -1;
    int keep = 0; // 用於 NMS 過程中標記是否保留此框
};

struct BBox
{
    cv::Mat orinImage;
    cv::Mat resizeImage;
    cv::cuda::GpuMat gpuInputImage;
    cv::Size modelInsize;
    std::vector<cv::Rect> rect;
    std::vector<int> indices;
    std::vector<int> classId;
    std::vector<float> score;
    Pad pad;
    yolocfg cfg;
};
extern std::vector<BBox> Bboxes;

struct caltime
{
    int warmtimes = 30;
    int inferencetimes = 180;
    std::vector<float> preProcesstime;
    std::vector<float> inferencetime;
    std::vector<float> postProcesstime;
    std::vector<float> totaltime;
};
extern caltime calTime;
#endif // STRUCT_TYPE_H
