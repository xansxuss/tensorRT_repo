#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "logger.h"
#include "infer.h"
#include "struct_type.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <climits>
#include "yamlParser.h"
#include "setEnv.h"
#include <thread>             // for std::this_thread
#include <condition_variable> // for std::condition_variable
#include <chrono>             // for std::chrono::milliseconds
#include <mutex>              // for std::mutex, std::unique_lock
#include <CLI/CLI.hpp>
#include <unordered_set>
#include <filesystem>
#include <time.h>

using namespace nvinfer1;
std::vector<BBox> Bboxes;
std::vector<BatchBBox> BatchBBoxes;
std::vector<Binding> mBindings;
configStruct configstruct;
std::condition_variable cvFrameavailable;
std::mutex mtxFrame;
BBox bbox;
BatchBBox batchbbox;

// std::vector<float> costtimes;

// std::deque<BBox> BboxesDeque;
// std::deque<BBox> inferDeque;

int main(int argc, char **argv)
{
    CLI::App app{"set arge"};
    auto customLogger = customLogger::getInstance();
    // const std::string configPath = "/workspaces_data/repo/NVIDIA/TensorRT/infer/config/config.yaml";
    std::string configPath;
    app.add_option("-c,--config", configPath, "set configure file")->check(CLI::ExistingFile)->required(); // 強制使用者一定要輸入並且檢查檔案存在
    CLI11_PARSE(app, argc, argv);
    customLogger->debug("set configure file path:{}", configPath);
    if (std::filesystem::path(configPath).extension() != ".yaml")
    {
        customLogger->critical("Error: Invalid config file");
        std::exit(EXIT_FAILURE);
    }
    yamlParser yamlparser;
    yamlparser.parseConfig(configPath, bbox, configstruct);

    customLogger->debug("Starting the application...");
    cv::Mat image = cv::imread(configstruct.imagePath);
    // customLogger->debug("Starting base inference...");
    batchYoloInfer infer(configstruct.enginePath);

    unsigned char *ImagePtr = nullptr;

    customLogger->debug("calulating single image bytes");
    size_t singleimageBytes = image.rows * image.cols * mBindings[0].C * sizeof(unsigned char);
    size_t imageSize = image.rows * image.cols * mBindings[0].C;
    customLogger->debug("single image bytes : {}", singleimageBytes);
    customLogger->debug("allocating memory for batch input image");
    ImagePtr = (unsigned char *)malloc(singleimageBytes * mBindings[0].N);
    while (true)
    {

        customLogger->debug("copying image data to batch input image");
        for (size_t i = 0; i < mBindings[0].N; ++i)
        {
            customLogger->debug("copying image data to batch input image, index: {}", i);
            memcpy(ImagePtr + i * imageSize, image.data, singleimageBytes);
        }
        cv::Mat Image;
        customLogger->debug("displaying batch input images");
        for (size_t i = 0; i < mBindings[0].N; ++i)
        {
            Image = cv::Mat(image.rows, image.cols, CV_8UC3, ImagePtr + i * imageSize);
            bbox.orinImage = Image;
            cv::namedWindow("batch image copy test", cv::WINDOW_NORMAL);
            cv::resizeWindow("batch image copy test", image.rows, image.cols);
            cv::setWindowTitle("batch image copy test", "batch image copy test " + std::to_string(i));
            cv::imshow("batch image copy test", bbox.orinImage);
            if (cv::waitKey(1) == 27) // 按下 ESC 鍵退出
            {
                customLogger->info("ESC key pressed, exiting...");
                break;
            }
        }
        // if (getImshowFlag("IMSHOW_FLAG"))
        // {
        //     cv::imshow("GPU Decoded Frame", bbox.orinImage);
        //     if (cv::waitKey(1) == 27) // 按下 ESC 鍵退出
        //         break;
        // }
    }

    return 0;
}
