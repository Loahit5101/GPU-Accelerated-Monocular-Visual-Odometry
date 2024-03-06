cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE BOOL "Cuda compiler (nvcc)")

project(CUDA_VO LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package( OpenCV 4.2.0 REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )
include_directories(include)

add_executable(benchmark src/benchmark.cpp src/GPU/ORB.cu src/CPU/ORB_CPU.cpp )
target_link_libraries( benchmark ${OpenCV_LIBS} )

add_executable(main src/main.cpp src/GPU/VO_class.cpp src/GPU/ORB.cu src/CPU/ORB_CPU.cpp )
target_link_libraries( main ${OpenCV_LIBS} )


add_executable(bf src/GPU/bf_matcher.cu src/GPU/ORB.cu)
target_link_libraries( bf ${OpenCV_LIBS} )


#cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..
