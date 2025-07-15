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

using namespace nvinfer1;
// const std::string engineFile = "/home/eray/repo/NVIDIA/TensorRT/infer/model/yolov8n.engine";
std::vector<BBox> Bboxes;
std::vector<Binding> mBindings;
configStruct configstruct;
std::condition_variable cvFrameavailable;
std::mutex mtxFrame;
BBox bbox;

std::deque<BBox> BboxesDeque;
std::deque<BBox> inferDeque;

// ==== Logger 實作 ====
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void getRTSPframe()
{

    // GStreamer Pipeline - NVIDIA H264 Decode with appsink
    std::string pipeline =
        "rtspsrc location=rtsp://eray80661707:80661707@192.168.33.92:554/stream1 latency=500 is-live=true ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! videoconvert ! video/x-raw, format=BGR ! appsink";
    customLogger::getInstance()->debug("Using GStreamer pipeline: {}", pipeline);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        customLogger::getInstance()->error("Failed to open video stream with GStreamer pipeline: {}", pipeline);
        return;
    }
    customLogger::getInstance()->debug("Video stream opened successfully with GStreamer pipeline: {}", pipeline);

    cv::Mat frame;
    BBox bbox;
    while (cap.read(frame))
    {
        if (frame.empty())
        {
            customLogger::getInstance()->error("Received empty frame from video stream, exiting...");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        bbox.orinImage = frame;
        std::lock_guard<std::mutex> lock(mtxFrame);
        BboxesDeque.emplace_back(bbox);
        cvFrameavailable.notify_one();
        // customLogger::getInstance()->debug("Read frame from video stream, size: {}", BboxesDeque.size());

        // customLogger::getInstance()->debug("Read frame from video stream, size: {}x{}", frame.cols, frame.rows);
        // cv::namedWindow("GPU Decoded Frame", cv::WINDOW_NORMAL);
        // cv::imshow("GPU Decoded Frame", BboxesDeque.front().orinImage);
        // BboxesDeque.pop_front();
        // if (cv::waitKey(1) == 27)
        //     break;
    }
    // cv::destroyAllWindows();
    cap.release();
}
void showRTSPframe()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    customLogger::getInstance()->debug("Starting to show frames from deque...");
    int count = 0;
    BBox bbox;
    baseInfer infer(configstruct.enginePath);
    while (true)
    {
        std::unique_lock<std::mutex> lock(mtxFrame); // ✅ 用 unique_lock
        cvFrameavailable.wait(lock, []
                              { return !BboxesDeque.empty(); });
        if (BboxesDeque.empty())
        {
            customLogger::getInstance()->error("No frames in deque to show.");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        bbox = BboxesDeque.front();
        BboxesDeque.pop_front();
        inferDeque.emplace_back(bbox);
        inferDeque.pop_front();
        lock.unlock();
        infer.baseInferenceGPU(bbox);
        cv::Mat frame = bbox.orinImage;
        // cv::imshow("GPU Decoded Frame", frame);
        // std::ostringstream oss;
        // oss << "GPU Decoded Frame:" << std::setfill('0') << std::setw(4) << count++ << ".jpg";
        // std::string filename = oss.str();

        // if (count % 1000 == 0)
        // {
        //     cv::imwrite(filename, bbox.orinImage);
        // }
        // count++;
        cv::namedWindow("GPU Decoded Frame", cv::WINDOW_NORMAL);
        cv::imshow("GPU Decoded Frame", bbox.orinImage);

        customLogger::getInstance()->debug("pop element from BboxesDeque, size: {}", BboxesDeque.size());
        customLogger::getInstance()->debug("pop element from inferDeque, size: {}", inferDeque.size());
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (cv::waitKey(1) == 27)
            break;
    }
    cv::destroyAllWindows();
}

int main()
{
    const std::string configPath = "/home/eray/repo/NVIDIA/TensorRT/infer/config/config.yaml";
    yamlParser yamlparser;
    yamlparser.parseConfig(configPath, bbox, configstruct);
    auto customLogger = customLogger::getInstance();
    customLogger->debug("Starting the application...");

    // cv::Mat image = cv::imread("/home/eray/repo/datasets/coco/000000005060.jpg");    
    cv::Mat image = cv::imread(configstruct.imagePath);
    // std::thread rtspThread(getRTSPframe);

    // if (BboxesDeque.empty())
    // {
    //     customLogger->debug("Waiting for frames to be added to deque...");
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // }
    // std::thread showThread(showRTSPframe);
    // // rtspThread.join();
    // showThread.join();

    // customLogger->debug("Finished showing frames from deque.");

    // std::string pipeline =
    //         "rtspsrc location=rtsp://eray8061707:80661707@192.168.33.92:554/stream1 latency=500 is-live=true ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! videoconvert ! video/x-raw, format=BGR ! appsink";
    //     customLogger::getInstance()->debug("Using GStreamer pipeline: {}", pipeline);
    //     cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    //     if (!cap.isOpened())
    //     {
    //         customLogger::getInstance()->error("Failed to open video stream with GStreamer pipeline: {}", pipeline);
    //         return 0;
    //     }
    //     customLogger::getInstance()->debug("Video stream opened successfully with GStreamer pipeline: {}", pipeline);

    //     cv::Mat frame;
    // BBox bbox;
    customLogger->debug("Starting base inference...");
    baseInfer infer(configstruct.enginePath);
    customLogger->debug("Base inference initialized with engine file: {}", configstruct.enginePath);
    int processcount = 0;
    // while (cap.read(frame))
    // {
    //     if (frame.empty())
    //     {
    //         customLogger::getInstance()->error("Received empty frame from video stream, exiting...");
    //         std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //     }

    //     bbox.orinImage = frame;
    //     // BboxesDeque.emplace_back(bbox);
    //     // customLogger::getInstance()->debug("Read frame from video stream, size: {}", BboxesDeque.size());
    //     infer.baseInferenceGPU(bbox);
    //     // customLogger::getInstance()->debug("Read frame from video stream, size: {}x{}", frame.cols, frame.rows);
    //     // BboxesDeque.pop_front();

    //     // if (cv::waitKey(1) == 27)
    //     //     break;
    // }
    // cv::Mat frame;
    // bbox.orinImage = image;
    // infer.baseInferenceGPU(bbox);
    
    while (true)
    {
        customLogger->info("Process count: {}", processcount);
        bbox.orinImage = image;
        infer.baseInferenceGPU(bbox);
        // frame = bbox.orinImage;
        cv::namedWindow("GPU Decoded Frame", cv::WINDOW_NORMAL);
        cv::resizeWindow("GPU Decoded Frame", 640, 640);
        cv::imshow("GPU Decoded Frame", bbox.orinImage);
        bbox.orinImage.release();
        bbox.resizeImage.release();
        bbox.rect.clear();
        bbox.indices.clear();
        bbox.classId.clear();
        bbox.score.clear();
        if (processcount == INT_MAX){
            processcount = 1;
        }
        processcount++;
        if (cv::waitKey(1) == 27)
            break;
    }

    cv::destroyAllWindows();
    // cap.release();
    customLogger->debug("Application finished successfully.");
    return 0;
}
