cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE BOOL "Cuda compiler (nvcc)")

project(CUDA_VO LANGUAGES CXX CUDA)

find_package( OpenCV 4.2.0 REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )

add_executable( vo VO_openCV.cpp )
target_include_directories(vo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(vocpu VO_CPU.cpp ORB_CPU.cpp )
target_include_directories(vocpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


add_executable(bf ORB_CPU.cpp bf_matcher.cu)

target_link_libraries( vo ${OpenCV_LIBS} )
target_link_libraries( vocpu ${OpenCV_LIBS} )
target_link_libraries( bf ${OpenCV_LIBS} )