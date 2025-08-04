#include "yamlParser.h"

yamlParser::yamlParser()
{
    // Constructor
}
yamlParser::~yamlParser()
{
    // Destructor
}
void yamlParser::parseConfig(const std::string &configFilePath, BBox &bbox, configStruct &configstruct)
{
    // Parse YAML config file
    try
    {
        config = YAML::LoadFile(configFilePath);
    }
    catch (const std::exception &e)
    {
        customLogger::getInstance()->error("Failed to load YAML file: {}", e.what());
        return;
    }
    // Read config values
    // parse imagePath
    configstruct.imagePath = getScalarOrDefault<std::string>(
        config["media"], "imagePath", "./media/input/000000005060.jpg", "media.imagePath");
    customLogger::getInstance()->info("Image path: {}", configstruct.imagePath);
    // parse model path
    configstruct.enginePath = getScalarOrDefault<std::string>(
        config["model"], "modelPath", "./model/yolov8n.engine", "model.modelPath");
    customLogger::getInstance()->info("Model path: {}", configstruct.enginePath);
    // parse confThreshold
    bbox.cfg.confThreshold = getScalarOrDefault<float>(
        config["model"], "confThreshold", 0.5f, "model.confThreshold");
    customLogger::getInstance()->info("Confidence threshold: {}", bbox.cfg.confThreshold);
    // parse nmsThreshold
    bbox.cfg.nmsThreshold = getScalarOrDefault<float>(
        config["model"], "nmsThreshold", 0.45f, "model.nmsThreshold");
    customLogger::getInstance()->info("NMS threshold: {}", bbox.cfg.nmsThreshold);
    // parse batchSize
    bbox.cfg.iouThreshold = getScalarOrDefault<float>(
        config["model"], "iouThreshold", 0.3f, "model.iouThreshold");
    customLogger::getInstance()->info("IOU threshold: {}", bbox.cfg.iouThreshold);
    // parse savePath
    configstruct.savePath = getScalarOrDefault<std::string>(
        config["output"], "outputPath", "./results/", "output.outputPath");
    customLogger::getInstance()->info("Output path: {}", configstruct.savePath);
    // parse classNames
    const auto &classNamesNode = config["detection"]["classNames"];
    if (!classNamesNode || !classNamesNode.IsMap())
    {
        customLogger::getInstance()->warn("detection.classNames missing or not a map.");
    }
    else
    {
        for (const auto &pair : classNamesNode)
        {
            try
            {
                // 強制 key 是 int、value 是 string
                if (!pair.first.IsScalar() || !pair.second.IsScalar())
                {
                    customLogger::getInstance()->warn("Invalid classNames entry: key/value not scalar");
                    continue;
                }

                int index = pair.first.as<int>();
                std::string className = pair.second.as<std::string>();
                configstruct.classNames[index] = className;
                customLogger::getInstance()->info("Class {}: {}", index, className);
            }
            catch (const std::exception &e)
            {
                customLogger::getInstance()->warn("Failed to parse classNames entry: {}", e.what());
            }
        }

        if (configstruct.classNames.empty())
        {
            customLogger::getInstance()->warn("classNames map parsed but no valid entries found.");
        }
    }
}
