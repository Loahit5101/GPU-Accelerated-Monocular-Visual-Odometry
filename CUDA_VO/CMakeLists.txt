cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE BOOL "Cuda compiler (nvcc)")

project(CUDA_VO LANGUAGES CXX CUDA)

find_package( OpenCV 4.2.0 REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )
include_directories(include)

add_executable( vo src/CPU/VO_openCV.cpp )
target_include_directories(vo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(vocpu src/CPU/VO_CPU.cpp src/CPU/ORB_CPU.cpp )
target_include_directories(vocpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bf src/CPU/ORB_CPU.cpp src/GPU/bf_matcher.cu)


add_executable(fast_test src/GPU/fast_test.cpp src/GPU/fast.cu)


target_link_libraries( vo ${OpenCV_LIBS} )
target_link_libraries( vocpu ${OpenCV_LIBS} )
target_link_libraries( bf ${OpenCV_LIBS} )
target_link_libraries( fast_test ${OpenCV_LIBS} )