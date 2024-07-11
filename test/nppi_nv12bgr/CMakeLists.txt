cmake_minimum_required(VERSION 3.10)
project(nppi_test)

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
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()
################################

# 代码编译
################################
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
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
  ${CUDA_npp_LIBRARY}
)