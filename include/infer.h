#ifndef INFER_H
#define INFER_H
#pragma once

#define LIBRARY_VERSION_MAJOR 0
#define LIBRARY_VERSION_MINOR 1
#define LIBRARY_VERSION_PATCH 0

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include "logger.h"
#include "struct_type.h"
#include "imageProcess.h"

class TRTLogger : public nvinfer1::ILogger
{
public:
    explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= reportableSeverity)
        {
            std::cerr << "[TensorRT][" << severityToStr(severity) << "] " << msg << std::endl;
        }
    }

private:
    Severity reportableSeverity;

    const char *severityToStr(Severity s) const
    {
        switch (s)
        {
        case Severity::kINTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case Severity::kERROR:
            return "ERROR";
        case Severity::kWARNING:
            return "WARNING";
        case Severity::kINFO:
            return "INFO";
        case Severity::kVERBOSE:
            return "VERBOSE";
        default:
            return "UNKNOWN";
        }
    }
};

class baseInfer
{
    // base virtual class for inference
    // 這個類別是用來定義推論的基礎介面
    // 其他推論類別可以繼承這個類別並實作具體的推論邏輯
    // protected:
public:
    // virtual constructor and destructor
    explicit baseInfer(const std::string &enginePath);
    virtual ~baseInfer();
    // virtual init function to initialize the inference engine
    virtual void init(const std::string &enginePath);
    // virtual function to run inference
    // load TensorRT engine from file
    virtual void loadEngine(const std::string &enginePath, std::vector<char> &engineData);
    // virtual function to run inference
    virtual void baseInference(BBox &Bbox);
    virtual void baseInferenceGPU(BBox &Bbox);

private:
    // TrnsorRT Logger
    TRTLogger mLogger;
    // TrnsorRT Runtime
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    /// TensorRT Engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    /// TensorRT Execution Context
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    /// TrnsorRT IO mode
    std::shared_ptr<nvinfer1::TensorIOMode> mIOMode;

    std::unique_ptr<yoloPreprocess> mPreprocess;
    std::unique_ptr<yoloPreprocessGPU> mPreprocessGPU;
    std::unique_ptr<yoloPostprocess> mPostprocess;
    std::unique_ptr<yoloPostprocessGPU> mPostprocessGPU;
    virtual void allocateBindings(std::vector<Binding> &mBindings);

};

class YoloInfer
{
    // public 是提供 API；private 是封裝內部細節；protected 是給繼承者的介面。
public:
    /// 基礎推論類別
    /// 建構子
    YoloInfer(const std::string &enginePath);
    /// 解構子
    ~YoloInfer();
    /// 初始化推論引擎（如設定 logger、stream、memory 等）
    void init(const std::string &enginePath);
    /// 從 TensorRT Engine 檔案載入模型
    /// @param engine_path Engine 檔案路徑
    void loadEngine(const std::string &enginePath, std::vector<char> &engineData);
    void YoloInference(BBox &Bbox);
    void YoloInferenceGPU(BBox &Bbox);
    // void YoloInferenceGPUbatch(BBoxBatch &BBoxBatch);

protected:
    /// TensorRT Logger
    TRTLogger mLogger;
    /// TensorRT Runtime
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    /// TensorRT Engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    /// TensorRT Execution Context
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    /// TrnsorRT IO mode
    std::shared_ptr<nvinfer1::TensorIOMode> mIOMode;

    std::unique_ptr<yoloPreprocessGPU> mPreProcessGPU;
    std::unique_ptr<yoloPostprocess> mPostProcess;
    std::unique_ptr<yoloPostprocessGPU> mPostProcessGPU;
    void allocateBindings(std::vector<Binding> &mBindings);

private:
};

#endif