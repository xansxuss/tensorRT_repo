#ifndef YAML_PARSE_H
#define YAML_PARSE_H
#pragma once
#include <yaml-cpp/yaml.h>
#include <string>
#include "struct_type.h"
#include "logger.h"

/**
 * @brief 從 YAML Node 中安全讀取 scalar 值，若缺少或錯誤則回傳預設值
 *
 * @tparam T             欲轉型的目標類型（例如 std::string, int, float）
 * @param parent         YAML::Node 的父節點（可為 config["media"] 等）
 * @param key            欄位名稱
 * @param defaultValue   預設值
 * @param fullKeyName    用於 logger 的完整 key 字串（例如 "model.confThreshold"）
 * @return T             成功則為原始值，失敗則為 defaultValue
 */
template <typename T>
T getScalarOrDefault(const YAML::Node &parent, const std::string &key, const T &defaultValue, const std::string &fullKeyName)
{
    auto logger = customLogger::getInstance();

    if (!parent || !parent[key])
    {
        logger->warn("Missing key '{}', using default: {}", fullKeyName, defaultValue);
        return defaultValue;
    }

    try
    {
        return parent[key].as<T>();
    }
    catch (const std::exception &e)
    {
        logger->warn("Failed to parse key '{}': {}, using default: {}", fullKeyName, e.what(), defaultValue);
        return defaultValue;
    }
}

class yamlParser
{
public:
    yamlParser();
    ~yamlParser();
    void parseConfig(const std::string &configFilePath, BBox &bbox, configStruct &configstruct);

private:
    YAML::Node config;
};

#endif // YAML_PARSE_H