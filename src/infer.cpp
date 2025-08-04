#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "infer.h"
// #include "struct_type.h"
// #include "imageProcess.h"

// ======= 工具函數 =======
inline void checkCuda(cudaError_t result, const char *msg = "CUDA Error")
{
    if (result != cudaSuccess)
    {
        customLogger::getInstance()->error("CUDA Error: {}", cudaGetErrorString(result));
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(result));
    }
}

// ======= RAII CudaStream (move-only, priority-aware) =======
class CudaStream
{
public:
    // 建構：指定 flags 與優先權
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking, int priority = 0)
    {
        checkCuda(cudaStreamCreateWithPriority(&stream_, flags, priority), "cudaStreamCreateWithPriority failed");
    }

    // Move constructor
    CudaStream(CudaStream &&other) noexcept : stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }

    // Move assignment
    CudaStream &operator=(CudaStream &&other) noexcept
    {
        if (this != &other)
        {
            destroy();
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    // 禁止複製
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    ~CudaStream()
    {
        destroy();
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

    void synchronize() const
    {
        checkCuda(cudaStreamSynchronize(stream_), "Stream sync failed");
    }

private:
    void destroy()
    {
        if (stream_)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
    cudaStream_t stream_ = nullptr;
};

/// 基礎推論類別
/// 建構子實做
baseInfer::baseInfer(const std::string &enginePath)
{
    // constructor
    customLogger::getInstance()->debug("baseInfer constructor called with enginePath: {}", enginePath);
    init(enginePath);
    customLogger::getInstance()->debug("baseInfer constructor completed");
    mPreProcessGPU = std::make_unique<yoloPreprocessGPU>();
    mPostProcess = std::make_unique<yoloPostprocess>();
    mPostProcessGPU = std::make_unique<yoloPostprocessGPU>();
    customLogger::getInstance()->debug("baseInfer initialized with preProcessGPU and postProcess");
};

baseInfer::~baseInfer() {
    // Deconstructor
};

void baseInfer::init(const std::string &enginePath)
{
    mBindings.clear();
    std::vector<char> engineData;
    loadEngine(enginePath, engineData);
    if (engineData.empty())
    {
        customLogger::getInstance()->critical("Failed to read engine file: {}", enginePath);
        throw std::runtime_error("Failed to read engine file: " + enginePath);
        return;
    }
    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    if (!mRuntime)
    {
        throw std::runtime_error("createInferRuntime failed");
        customLogger::getInstance()->critical("createInferRuntime failed");
        return;
    }
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));

    if (!mEngine)
    {
        throw std::runtime_error("deserializeCudaEngine failed");
        customLogger::getInstance()->critical("deserializeCudaEngine failed");
        return;
    }
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext)
    {
        throw std::runtime_error("createExecutionContext failed");
        customLogger::getInstance()->critical("createExecutionContext failed");
        return;
    }
    customLogger::getInstance()->debug("Engine loaded successfully");
    allocateBindings(mBindings); // batch_size = 1
    customLogger::getInstance()->debug("Bindings allocated successfully");
    customLogger::getInstance()->debug("input N:{}", mBindings[0].N);
    customLogger::getInstance()->debug("input C:{}", mBindings[0].C);
    customLogger::getInstance()->debug("input H:{}", mBindings[0].H);
    customLogger::getInstance()->debug("input W:{}", mBindings[0].W);
    customLogger::getInstance()->debug("input Bindings dims: {}", mBindings[0].dims);
    customLogger::getInstance()->debug("input Bindings name: {}", mBindings[0].name);
    customLogger::getInstance()->debug("input Bindings dtype: {}", mBindings[0].dtype);
    customLogger::getInstance()->debug("input Bindings is_input: {}", mBindings[0].is_input);
    customLogger::getInstance()->debug("output Bindings N: {}", mBindings[1].N);
    customLogger::getInstance()->debug("output Bindings C: {}", mBindings[1].C);
    customLogger::getInstance()->debug("output Bindings H: {}", mBindings[1].H);
    customLogger::getInstance()->debug("output Bindings W: {}", mBindings[1].W);
    customLogger::getInstance()->debug("output Bindings dims: {}", mBindings[1].dims);
    customLogger::getInstance()->debug("output Bindings name: {}", mBindings[1].name);
    customLogger::getInstance()->debug("output Bindings dtype: {}", mBindings[1].dtype);
    customLogger::getInstance()->debug("output Bindings is_input: {}", mBindings[1].is_input);
};
void baseInfer::loadEngine(const std::string &enginePath, std::vector<char> &engineData)
{
    // 讀取引擎檔案
    std::ifstream engineFile(enginePath, std::ios::binary | std::ios::ate);
    if (!engineFile)
    {
        customLogger::getInstance()->critical("Failed to open engine file: {}", enginePath);
        throw std::runtime_error("Failed to open engine file: " + enginePath);
        return;
    }
    std::streamsize size = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    engineData.resize(size);
    engineFile.read(engineData.data(), size);
    if (!engineFile)
    {
        customLogger::getInstance()->critical("Failed to read engine file: {}", enginePath);
        throw std::runtime_error("Failed to open engine file: " + enginePath);
        return;
    }
}

void baseInfer::allocateBindings(std::vector<Binding> &mBindings)
{
    int nbTensors = mEngine->getNbIOTensors();
    // std::cout << "Number of I/O tensors: " << nbTensors << std::endl;
    for (int i = 0; i < nbTensors; ++i)
    {
        Binding b;
        b.name = mEngine->getIOTensorName(i);
        // if ((mEngine->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT ? "[Input]" : "[Output]") == "[Input]")
        // {
        //     b.is_input = 1;
        // }
        b.is_input = (mEngine->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT);
        
        b.dtype = mEngine->getTensorDataType(b.name.c_str());
        b.dims = mEngine->getTensorShape(b.name.c_str());
        b.N = b.dims.d[0];
        b.C = b.dims.d[1];
        b.H = b.dims.d[2];
        b.W = b.dims.d[3];
        // customLogger::getInstance()->debug("Binding N: {}", b.N);
        // customLogger::getInstance()->debug("Binding C: {}", b.C);
        // customLogger::getInstance()->debug("Binding H: {}", b.H);
        // customLogger::getInstance()->debug("Binding W: {}", b.W);
        size_t elem_count = 1;
        for (int j = 0; j < b.dims.nbDims; ++j)
        {
            elem_count *= b.dims.d[j];
        }
        size_t type_size = 0;
        switch (b.dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            type_size = 4;
            break;
        case nvinfer1::DataType::kHALF:
            type_size = 2;
            break;
        case nvinfer1::DataType::kINT8:
            type_size = 1;
            break;
        case nvinfer1::DataType::kINT32:
            type_size = 4;
            break;
        default:
            customLogger::getInstance()->error("Unsupported data type.");
            throw std::runtime_error("Unsupported data type.");
        }
        b.size = elem_count * type_size;
        cudaMalloc(&b.device_ptr, b.size);   // 分配 device memory
        // cudaMallocHost(&b.host_ptr, b.size); // 分配 pinned host memory
        cudaHostAlloc(&b.host_ptr, b.size,cudaHostAllocDefault);
        mBindings.push_back(b);
    }
};
void baseInfer::baseInference(BBox &Bbox)
{
    customLogger::getInstance()->debug("do baseInference");
    // customLogger::getInstance()->debug("input image size hight: {}, width: {}", Bbox.orinImage.rows, Bbox.orinImage.cols);
    Bbox.modelInsize = cv::Size(mBindings[0].W, mBindings[0].H);
    std::vector<float> nchw(mBindings[0].N * mBindings[0].C * mBindings[0].H * mBindings[0].W);

    yoloPreprocess preProcess;
    preProcess.run(Bbox, nchw);

    void *blob = reinterpret_cast<void *>(nchw.data());
    mBindings[0].host_ptr = blob; // 將 host_ptr 指向 nchw 的資料
    CudaStream stream;
    cudaMemcpyAsync(mBindings[0].device_ptr, mBindings[0].host_ptr, mBindings[0].size, cudaMemcpyHostToDevice, stream.get());
    mContext->setTensorAddress(mBindings[0].name.c_str(), mBindings[0].device_ptr);
    mContext->setTensorAddress(mBindings[1].name.c_str(), mBindings[1].device_ptr);
    bool ok = mContext->enqueueV3(stream.get());
    if (!ok)
    {
        customLogger::getInstance()->error("TensorRT enqueueV3 failed");
    }

    cudaMemcpyAsync(mBindings[1].host_ptr, mBindings[1].device_ptr, mBindings[1].size, cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    // float *outputData = static_cast<float *>(mBindings[1].host_ptr);
    // customLogger::getInstance()->debug("mBindings[1].size / sizeof(float) size:{}", mBindings[1].size / sizeof(float));
    // for (int i = 0; i < mBindings[1].size / sizeof(float); i++)
    // {
    //     customLogger::getInstance()->debug("outputData[{}]: {}", i, outputData[i]);
    // }
    yoloPostprocess yoloPostprocess;
    yoloPostprocess.run(Bbox);
    customLogger::getInstance()->debug("do baseInference done");
    for (int i = 0; i < Bbox.indices.size(); i++)
    {
        customLogger::getInstance()->debug("Bbox.rect[{}] x: {}, y: {}, w: {}, h: {}", i, Bbox.rect[i].x, Bbox.rect[i].y, Bbox.rect[i].width, Bbox.rect[i].height);
        // cv::rectangle(Bbox.orinImage, Bbox.rect[i], cv::Scalar(0, 0, 255), 2);
        // cv::rectangle(Bbox.resizeImage, Bbox.rect[i], cv::Scalar(0, 0, 255), 2);
    }
    // cv::imwrite("result.jpg", Bbox.orinImage);
};

void baseInfer::baseInferenceGPU(BBox &Bbox)
{

    CudaStream stream;
    customLogger::getInstance()->debug("do baseInferenceGPU");
    customLogger::getInstance()->debug("mBindings[0].w:{},mBindings[0].h:{}", mBindings[0].W, mBindings[0].H);
    Bbox.modelInsize = cv::Size(mBindings[0].W, mBindings[0].H);
    customLogger::getInstance()->debug("model input size: {}, {}", Bbox.modelInsize.width, Bbox.modelInsize.height);
    customLogger::getInstance()->debug("yolo preprocess GPU start");
    // yoloPreprocessGPU preProcessGPU;
    
    // auto pros = std::chrono::high_resolution_clock::now();
    
    mPreProcessGPU->run(Bbox);
    
    // auto proe = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> proc = proe - pros;
    // customLogger::getInstance()->info("pro cost time : {}", proc.count());
    // customLogger::getInstance()->info("pro FPS : {}", 1 / proc.count());
    
    customLogger::getInstance()->debug("yolo preprocess GPU end");

    // input image from GPU to CPU format cv::Mat
    // std::vector<float> nchw(640 * 640 * 3);
    // std::vector<float> nhwc(640 * 640 * 3);
    // cudaMemcpyAsync(nchw.data(), mBindings[0].device_ptr, 640*640*3*sizeof(float), cudaMemcpyDeviceToHost);
    // // customLogger::getInstance()->debug("mBindings[0].size / sizeof(float

    // for (int i = 0; i < 640; ++i)
    // {
    //     for (int j = 0; j < 640; ++j)
    //     {
    //         int hwcindex = (i * 640 + j) * 3; // 每個像素有3個通道
    //         int chwindex = (i * 640 + j);
    //         nhwc[hwcindex + 0] = nchw[chwindex + 0 * 640 * 640]; // R
    //         nhwc[hwcindex + 1] = nchw[chwindex + 1 * 640 * 640]; // G
    //         nhwc[hwcindex + 2] = nchw[chwindex + 2 * 640 * 640]; // B
    //     }
    // }

    // cv::Mat inputImage(640, 640, CV_32FC3, nhwc.data());
    // cv::Mat inputImageInt;
    // inputImage.convertTo(inputImageInt, CV_8UC3, 255.0);
    // cv::namedWindow("GPU Input Image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("GPU Input Image", 640, 640);
    // cv::imshow("GPU Input Image", inputImageInt);
    // cv::waitKey(0);

    mContext->setTensorAddress(mBindings[0].name.c_str(), mBindings[0].device_ptr);
    mContext->setTensorAddress(mBindings[1].name.c_str(), mBindings[1].device_ptr);
    
    // auto infers = std::chrono::high_resolution_clock::now();
    
    bool ok = mContext->enqueueV3(stream.get());
    
    // auto infere = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> inferc = infere - infers;
    // customLogger::getInstance()->info("infer cost time : {}", inferc.count());
    // customLogger::getInstance()->info("infer FPS : {}", 1 / inferc.count());
    
    if (!ok)
    {
        customLogger::getInstance()->critical("TensorRT enqueueV3 failed");
        return;
    }

    cudaMemcpyAsync(mBindings[1].host_ptr, mBindings[1].device_ptr, mBindings[1].size, cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    // // yoloPostprocessGPU yoloPostprocessGPU;
    // // yoloPostprocessGPU.run(Bbox, Bbox.pad);
    // yoloPostprocess yoloPostprocess;
    // mPostProcess->run(Bbox, Bbox.pad);
    
    // auto posts = std::chrono::high_resolution_clock::now();
    
    mPostProcessGPU->run(Bbox, Bbox.pad);
    
    // auto poste = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> postc = poste - posts;
    // customLogger::getInstance()->info("post cost time : {}", postc.count());
    // customLogger::getInstance()->info("post FPS : {}", 1 / postc.count());
    
    // // customLogger::getInstance()->debug("do baseInferenceGPU done");

    // cv::Mat resultImage = Bbox.orinImage.clone();
    // for (int i = 0; i < Bbox.indices.size(); i++)
    // {
    //     customLogger::getInstance()->debug("Bbox.rect[{}] x: {}, y: {}, w: {}, h: {}, id: {}", i, Bbox.rect[i].x, Bbox.rect[i].y, Bbox.rect[i].width, Bbox.rect[i].height, Bbox.classId[i]);
    //     char classIdStr[16];
    //     sprintf(classIdStr, "%d", Bbox.classId[Bbox.indices[i]]);

    //     cv::rectangle(resultImage, Bbox.rect[i], cv::Scalar(0, 0, 255), 2);
    //     // cv::rectangle(Bbox.resizeImage, Bbox.rect[i], cv::Scalar(0, 0, 255), 2);
    // }

    // cv::namedWindow("Result Image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Result Image", 640, 640);
    // cv::imshow("Result Image", resultImage);
}
