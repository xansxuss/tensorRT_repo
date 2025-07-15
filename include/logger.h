#ifndef LOGGER_H
#define LOGGER_H

#define LIBRARY_VERSION_MAJOR 0
#define LIBRARY_VERSION_MINOR 1
#define LIBRARY_VERSION_PATCH 0


#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <string>
#include <filesystem>
#include <opencv2/core/types.hpp> // 引入 cv::Size_<int> 類型定義
#include <spdlog/fmt/fmt.h>       // 引入 fmt 庫的格式化功能
#include <spdlog/sinks/stdout_color_sinks.h>
#include <NvInfer.h>

const char* dataTypeToStr(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:    return "kFLOAT";
        case nvinfer1::DataType::kHALF:     return "kHALF";
        case nvinfer1::DataType::kINT8:     return "kINT8";
        case nvinfer1::DataType::kINT32:    return "kINT32";
#if NV_TENSORRT_MAJOR >= 8
        case nvinfer1::DataType::kBOOL:     return "kBOOL";
#endif
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kFP8:      return "kFP8";
        case nvinfer1::DataType::kBF16:     return "kBF16";
#endif
        default:                            return "UNKNOWN";
    }
}

// custom formatter by opencv cv::Size
namespace fmt
{
    template <>
    struct formatter<cv::Size_<int>>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const cv::Size_<int> &s, FormatContext &ctx) const
        {
            // 以 "(width, height)" 形式輸出
            return format_to(ctx.out(), "(width:{}, height:{})", s.width, s.height);
        }
    };

    // custom formatter by opencv cv::Rect_<int>
    template <>
    struct formatter<cv::Rect_<int>>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const cv::Rect_<int> &r, FormatContext &ctx) const
        {
            return format_to(ctx.out(), "(x:{}, y:{}, width:{}, height:{})", r.x, r.y, r.width, r.height);
        }
    };
    // custom formatter by opencv cv::Rect_<float>
    template <>
    struct formatter<cv::Rect_<float>>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const cv::Rect_<float> &r, FormatContext &ctx) const
        {
            return format_to(ctx.out(), "(x:{:.3f}, y:{:.3f}, width:{:.3f}, height:{:.3f})", r.x, r.y, r.width, r.height);
        }
    };

    // custom formatter by opencv cv::Point3f
    template <>
    struct formatter<cv::Point3f>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const cv::Point3f &r, FormatContext &ctx) const
        {
            return format_to(ctx.out(), "(x:{:.4f}, y:{:.4f}, z:{:.4f})", r.x, r.y, r.z);
        }
    };

    // custom formatter by opencv cv::Point2i
    template <>
    struct formatter<cv::Point2i>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const cv::Point2i &r, FormatContext &ctx) const
        {
            return format_to(ctx.out(), "(x:{}, y:{})", r.x, r.y);
        }
    };

    // custom formatter by std::filesystem.string()
    template <>
    struct formatter<std::filesystem::path>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const std::filesystem::path &r, FormatContext &ctx) const
        {
            return format_to(ctx.out(), r.string());
        }
    };
    // custom formatter by size_t
    template <>
    struct formatter<size_t>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }

        // 定義格式化規則
        template <typename FormatContext>
        auto format(const size_t &r, FormatContext &ctx) const
        {
            return format_to(ctx.out(), "(size:{})", r);
        }
    };

    // custom formatter by nvinfer1::Dims
    template <>
    struct formatter<nvinfer1::Dims>
    {
        // 解析格式化字符串的函數
        template <typename ParseContext>
        constexpr auto parse(ParseContext &ctx)
        {
            // 這裡簡單跳過解析過程，直接返回
            return ctx.begin();
        }
        // 定義格式化規則
        template <typename FormatContext>
        auto format(const nvinfer1::Dims &r, FormatContext &ctx) const
        {
            std::string dims_str = "(";
            for (int i = 0; i < r.nbDims; ++i)
            {
                dims_str += std::to_string(r.d[i]);
                if (i < r.nbDims - 1)
                {
                    dims_str += ", ";
                }
            }
            dims_str += ")";
            return format_to(ctx.out(), "{}", dims_str);
        }
    };
    // custom formatter by nvinfer1::DataType
    template <>
    struct formatter<nvinfer1::DataType>
    {
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        template <typename FormatContext>
        auto format(nvinfer1::DataType type, FormatContext &ctx) const
        {
            return format_to(ctx.out(), "{}", dataTypeToStr(type));
        }
    };
}

class customLogger
{
public:
    // 獲取共享的 Logger 實例
    static std::shared_ptr<spdlog::logger> &getInstance();

    // 設置日志等級
    static void setLogLevel(spdlog::level::level_enum level);

    // 禁止拷貝和賦值
    customLogger(const customLogger &) = delete;
    customLogger &operator=(const customLogger &) = delete;

private:
    customLogger() = default;
    ~customLogger() = default;
};

#endif // LOGGER_H