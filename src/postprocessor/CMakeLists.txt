cmake_minimum_required(VERSION 3.10)
project(traffic_light_postprocessor)

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

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
################################

# 代码编译
################################
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/include
)

list(APPEND CODE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/src/BYTETracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/src/kalmanFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/src/lapjv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/src/STrack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/traffic_light_postprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
)

add_library(${PROJECT_NAME} SHARED
    ${CODE_SRC}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
)
##################################


# 安装库文件
##################################
install(TARGETS
  ${PROJECT_NAME}
  DESTINATION
  ${CMAKE_SOURCE_DIR}/lib/
)
####################################