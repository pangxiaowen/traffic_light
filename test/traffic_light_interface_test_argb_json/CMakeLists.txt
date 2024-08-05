cmake_minimum_required(VERSION 3.10)
project(traffic_light_interface_test)

# 编译配置
################################

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=c++14 -g -ggdb -O0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=c++14 -O2")
endif()
################################

# 代码编译
################################
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../interface
)

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
)

list(APPEND CODE_SRC
  main.cpp
)

add_executable(${PROJECT_NAME} 
    ${CODE_SRC}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  traffic_light
  traffic_light_interface
  traffic_light_preprocessor
  traffic_light_postprocessor
  traffic_light_detector
  boost_filesystem
)