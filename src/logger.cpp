#include <filesystem>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <algorithm> // for std::transform
#include <cctype>    // for ::tolower
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include "logger.h"

enum class LogLevel
{
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARN = 3,
        ERROR = 4,
        FATAL = 5,
        OFF = 6 // 不輸出任何 log
};

// 把字串轉成 LogLevel 枚舉
LogLevel parseLogLevel(std::string &levelStr)
{
        std::string lowerStr = levelStr;
        // // 將 levelStr 轉成小寫
        // std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(),
        //                [](unsigned char c)
        //                { return std::tolower(c); });
        if (lowerStr == "trace")
        {
                return LogLevel::TRACE;
        }
        else if (lowerStr == "debug")
        {
                return LogLevel::DEBUG;
        }
        else if (lowerStr == "info")
        {
                return LogLevel::INFO;
        }
        else if (lowerStr == "warn")
        {
                return LogLevel::WARN;
        }
        else if (lowerStr == "error")
        {
                return LogLevel::ERROR;
        }
        else if (lowerStr == "fatal")
        {
                return LogLevel::FATAL;
        }
        else
        {
                return LogLevel::OFF; // 預設 off
        }
}

void renameOldlog()
{
        namespace fs = std::filesystem;
        std::string oldlog = "logs/infer_tool.log";
        if (!fs::exists(oldlog))
                return;
        // 取得目前時間
        std::time_t t = std::time(nullptr);
        std::tm *tm_info = std::localtime(&t);

        char date_str[20];
        std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm_info);
        // 目標檔案名稱，例如 "2025-03-12_infer_tool.log"
        std::string new_name = "logs/" + std::string(date_str) + "_infer_tool.log";
        // 如果檔案已存在，就不覆蓋
        if (!fs::exists(new_name))
        {
                try
                {
                        fs::rename(oldlog, new_name);
                }
                catch (const fs::filesystem_error &e)
                {
                        std::cerr << "Error renaming log file: " << e.what() << std::endl;
                }
        }
}

// 定義靜態 Logger 實例
std::shared_ptr<spdlog::logger> &customLogger::getInstance()
{

        static std::shared_ptr<spdlog::logger> instance = []
        {
                const char *env_cstr = std::getenv("CUSTOMS_LEVEL_DEBUG");
                std::string envString = env_cstr ? env_cstr : "";

                std::cout << "Custom log level set to: " << envString << std::endl;

                LogLevel customloglevel = parseLogLevel(envString);
                std::cout << "Parsed custom log level: " << static_cast<int>(customloglevel) << std::endl;
                renameOldlog();
                // 創建終端輸出（彩色）
                auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [PID: %P]  [Console] [%^%l%$] %v");

                // // 每天生成新日誌
                // auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/infer_tool.log", 0, 0, false, 7);
                // daily_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [PID: %P]  [File] [%l] %v");

                // // 限制檔案大小（50MB），最多保留 5 個檔案
                // auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/infer_tool.log", 50 * 1024 * 1024, 5);
                // rotating_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [PID: %P]  [Rotating] [%l] %v");

                // 創建檔案輸出
                auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/infer_tool.log", true);
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [PID: %P]  [File] [%l] %v");

                // 組合多個輸出
                // std::vector<spdlog::sink_ptr> sinks{console_sink, rotating_sink};
                std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
                auto customLogger = std::make_shared<spdlog::logger>("infer_logger", sinks.begin(), sinks.end());

                // #ifdef CUSTOMS_LEVEL_DEBUG
                //                 customLogger->set_level(spdlog::level::debug);
                //                 // std::cout<<"debug"<<std::endl;
                // #elif CUSTOMS_LEVEL_INFO
                //                 customLogger->set_level(spdlog::level::info);
                //                 // std::cout<<"info"<<std::endl;
                // #elif CUSTOMS_LEVEL_WARN
                //                 customLogger->set_level(spdlog::level::warn);
                //                 // std::cout<<"warn"<<std::endl;
                // #else
                //                 customLogger->set_level(spdlog::level::err);
                //                 // std::cout<<"error"<<std::endl;

                // #endif
                //                 // logger->set_level(spdlog::level::off); // 默認關閉日誌
                //                 customLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [PID: %P]  [%l] %v");

                if (customloglevel == LogLevel::TRACE)
                {
                        customLogger->set_level(spdlog::level::trace);
                }
                else if (customloglevel == LogLevel::DEBUG)
                {
                        customLogger->set_level(spdlog::level::debug);
                }
                else if (customloglevel == LogLevel::INFO)
                {
                        customLogger->set_level(spdlog::level::info);
                }
                else if (customloglevel == LogLevel::WARN)
                {
                        customLogger->set_level(spdlog::level::warn);
                }
                else if (customloglevel == LogLevel::ERROR)
                {
                        customLogger->set_level(spdlog::level::err);
                }
                else if (customloglevel == LogLevel::FATAL)
                {
                        customLogger->set_level(spdlog::level::critical);
                }
                else
                {
                        customLogger->set_level(spdlog::level::off); // 不輸出任何 log
                }

                return customLogger;
        }();
        return instance;
}

// // 設置日志等級
// void Logger::setLogLevel(spdlog::level::level_enum level)
// {
//     getInstance()->set_level(level);
// }