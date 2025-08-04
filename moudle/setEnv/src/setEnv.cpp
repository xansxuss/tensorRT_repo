#include "setEnv.h"
#include <unordered_set>
#include <algorithm>
#include <cstdlib>

bool getImshowFlag(const std::string &value)
{
    const char *env_imshow = std::getenv(value.c_str());
    bool defaultValue = false;
    if (!env_imshow){return defaultValue;}
    static const std::unordered_set<std::string> truthySet = {
        "1", "true", "yes", "on", "enable"};
    std::string lower = env_imshow;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    return truthySet.count(lower) > 0;
}