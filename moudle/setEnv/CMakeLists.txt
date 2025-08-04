cmake_minimum_required(VERSION 3.20)
project(setEnvLib VERSION 1.0 LANGUAGES CXX)

# 可由使用者指定編譯類型（ON: shared, OFF: static）
option(SETENV_BUILD_SHARED "Build setEnv shared library" ON)

# 根據選項決定要建立哪種 lib
if(SETENV_BUILD_SHARED)
    add_library(setEnv SHARED src/setEnv.cpp)
else()
    add_library(setEnv STATIC src/setEnv.cpp)
endif()

# 統一名稱，無論是 .so 或 .a 都叫 libsetEnv.*
set_target_properties(setEnv PROPERTIES OUTPUT_NAME setEnv)

# 包含標頭路徑
target_include_directories(setEnv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 提供 namespace alias
add_library(SetEnv::SetEnv ALIAS setEnv)

# 安裝 lib（依據 shared or static）
install(TARGETS setEnv
    EXPORT SetEnvTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安裝 header
install(DIRECTORY include/ DESTINATION include)

# 將 target 輸出為 cmake config
install(EXPORT SetEnvTargets
    FILE SetEnvTargets.cmake
    NAMESPACE SetEnv::
    DESTINATION lib/cmake/SetEnv
)

# 產生 SetEnvConfig.cmake 與 SetEnvConfigVersion.cmake
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SetEnvConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetEnvConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SetEnvConfig.cmake"
    INSTALL_DESTINATION lib/cmake/SetEnv
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SetEnvConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SetEnvConfigVersion.cmake"
    DESTINATION lib/cmake/SetEnv
)
