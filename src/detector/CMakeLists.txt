cmake_minimum_required(VERSION 3.10)
project(traffic_light_detector)

# 编译配置
################################
# enable cuda
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=c++14 -g -ggdb")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=c++14 -O2")
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)
  # tensorrt
  include_directories(/home/pxw/project/TensorRT-8.5.2.2/include)
  link_directories(/home/pxw/project/TensorRT-8.5.2.2/lib)
endif()
################################

# 代码编译
################################
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

list(APPEND CODE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/tensorrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trt_preprocess.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/traffic_light_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/yoloPlugins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/yoloForward_nc.cu
)

# find_package(OpenCV REQUIRED)
add_library(${PROJECT_NAME} SHARED
    ${CODE_SRC}
)

target_link_libraries(${PROJECT_NAME}
    nvinfer
    nvinfer_plugin
    dl
    ${CUDA_LIBRARIES}
    ${CUDA_npp_LIBRARY}
    ${OpenCV_LIBS}
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION
  ${CMAKE_SOURCE_DIR}/lib/
)