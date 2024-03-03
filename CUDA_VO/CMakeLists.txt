cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE BOOL "Cuda compiler (nvcc)")

project(CUDA_VO LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package( OpenCV 4.2.0 REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )
include_directories(include)

add_executable(vocpu src/CPU/VO_CPU.cpp src/CPU/ORB_CPU.cpp )
target_include_directories(vocpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#add_executable(orb src/GPU/ORB.cu src/CPU/ORB_CPU.cpp )

add_executable(benchmark src/GPU/benchmark.cpp src/GPU/ORB.cu src/CPU/ORB_CPU.cpp )

add_executable(class src/GPU/VO_class.cpp src/GPU/ORB.cu src/CPU/ORB_CPU.cpp )


target_link_libraries( vocpu ${OpenCV_LIBS} )
#target_link_libraries( orb ${OpenCV_LIBS} )
target_link_libraries( benchmark ${OpenCV_LIBS} )
target_link_libraries( class ${OpenCV_LIBS} )


#cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..
