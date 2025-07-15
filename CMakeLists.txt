cmake_minimum_required(VERSION 3.20)
project(tensorrt_infer)

# target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
# # 設定 C++ 標準
set(CMAKE_CXX_STANDARD 20) # 或 14、17 等
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g ") # -fsanitize=address
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ") # -fsanitize=address
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

add_compile_options(-Wno-deprecated-declarations)


# 設定輸出目錄
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_CUDA_STANDARD 14)
enable_language(CUDA)
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_package(spdlog REQUIRED PATHS)
include_directories(${spdlog_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED PATHS)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# 手動指定 TensorRT 與 CUDA 的位置
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_ROOT})
set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
# include 路徑
include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/include
)

# lib 路徑
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${TENSORRT_ROOT}/lib
)

# 建立共享庫 logger
add_library(logger SHARED src/logger.cpp)
target_link_libraries(logger PRIVATE spdlog::spdlog ${OpenCV_LIBS})

add_library(yamlParser SHARED src/yamlParser.cpp)
target_link_libraries(yamlParser PRIVATE spdlog::spdlog ${OpenCV_LIBS} yaml-cpp)

# 建立共享庫 imageProcess
add_library(imageProcess SHARED src/imageProcess.cpp src/GPU_utils.cu)
target_link_libraries(imageProcess PRIVATE spdlog::spdlog ${OpenCV_LIBS} cudart)

# 建立共享庫 infer
add_library(infer SHARED src/infer.cpp)
target_link_libraries(infer PRIVATE spdlog::spdlog ${OpenCV_LIBS} nvinfer nvonnxparser cudart)



add_executable(${PROJECT_NAME} main.cpp)

# TensorRT 需要的 link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    cudart
    nvinfer
    nvonnxparser
    spdlog::spdlog
    yamlParser
    logger
    infer
    imageProcess
)

add_executable(cudadtest cudatest.cpp)

target_link_libraries(cudadtest
    ${OpenCV_LIBS}
    cudart
    nvinfer
    nvonnxparser
    spdlog::spdlog
    logger
    infer
    imageProcess
)