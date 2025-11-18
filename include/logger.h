#ifndef LOGGER_H
#define LOGGER_H

#define LIBRARY_VERSION_MAJOR 0
#define LIBRARY_VERSION_MINOR 1
#define LIBRARY_VERSION_PATCH 0

#include <memory>
#include <string>
#include <filesystem>

// spdlog / fmt (注意：formatter 必須在使用 fmt::basic_format_string 的 translation unit 之前可見)
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/bundled/ranges.h> // 如果你想直接格式化 std::vector<T>（或使用 fmt::join），需要這個 header

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <NvInfer.h>

const char* dataTypeToStr(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:    return "kFLOAT";
        case nvinfer1::DataType::kHALF:     return "kHALF";
        case nvinfer1::DataType::kINT8:     return "kINT8";
        case nvinfer1::DataType::kINT32:    return "kINT32";
#if NV_TENSORRT_MAJOR >= 7
        case nvinfer1::DataType::kINT64:    return "kINT64";
        case nvinfer1::DataType::kINT4:     return "kINT4";
#endif
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

// ---------- fmt formatters for OpenCV / nvInfer ----------
namespace fmt {

// Generic formatter for cv::Size_<T>
template <typename T>
struct formatter<cv::Size_<T>> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const cv::Size_<T> &s, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "(w:{}, h:{})",
                              static_cast<double>(s.width),
                              static_cast<double>(s.height));
    }
};

// Generic formatter for cv::Rect_<T>
template <typename T>
struct formatter<cv::Rect_<T>> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const cv::Rect_<T> &r, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(),
                              "(x:{:.3f}, y:{:.3f}, w:{:.3f}, h:{:.3f})",
                              static_cast<double>(r.x),
                              static_cast<double>(r.y),
                              static_cast<double>(r.width),
                              static_cast<double>(r.height));
    }
};

// Generic formatter for cv::Point_<T>
template <typename T>
struct formatter<cv::Point_<T>> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const cv::Point_<T> &p, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(),
                              "(x:{:.4f}, y:{:.4f})",
                              static_cast<double>(p.x),
                              static_cast<double>(p.y));
    }
};

// ✅ formatter for cv::MatStep （修正重點）
template <>
struct formatter<cv::MatStep> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const cv::MatStep &s, FormatContext &ctx) const {
        // 一般使用 s[0] 表示一列的 byte pitch
        // 若是多維資料，可自行調整顯示
        return fmt::format_to(ctx.out(), "step0:{} bytes", static_cast<size_t>(s[0]));
    }
};

// formatter for std::filesystem::path
template <>
struct formatter<std::filesystem::path> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const std::filesystem::path &p, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "{}", p.string());
    }
};

// formatter for nvinfer1::Dims
template <>
struct formatter<nvinfer1::Dims> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const nvinfer1::Dims &d, FormatContext &ctx) const {
        std::string s = "(";
        for (int i = 0; i < d.nbDims; ++i) {
            s += std::to_string(d.d[i]);
            if (i + 1 < d.nbDims) s += ", ";
        }
        s += ")";
        return fmt::format_to(ctx.out(), "{}", s);
    }
};

// formatter for nvinfer1::DataType
template <>
struct formatter<nvinfer1::DataType> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) { return ctx.begin(); }
    template <typename FormatContext>
    auto format(nvinfer1::DataType t, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "{}", dataTypeToStr(t));
    }
};

} // namespace fmt

// ---------- logger class ----------
class customLogger {
public:
    static std::shared_ptr<spdlog::logger> &getInstance();
    static void setLogLevel(spdlog::level::level_enum level);

    customLogger(const customLogger &) = delete;
    customLogger &operator=(const customLogger &) = delete;

private:
    customLogger() = default;
    ~customLogger() = default;
};

#endif // LOGGER_H
