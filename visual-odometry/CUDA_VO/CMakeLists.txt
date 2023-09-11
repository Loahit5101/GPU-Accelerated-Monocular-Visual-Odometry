cmake_minimum_required(VERSION 2.8)
project(CUDA_VO)
find_package( OpenCV 4.2.0 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( vo VO.cpp )
add_executable( fast FAST_cpu.cpp )
target_link_libraries( vo ${OpenCV_LIBS} )
target_link_libraries( fast ${OpenCV_LIBS} )
