cmake_minimum_required(VERSION 2.8)
project(CUDA_VO)
find_package( OpenCV 4.2.0 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( vo VO_openCV.cpp )

target_include_directories(vo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(orb ORB_CPU.cpp )


add_executable(vocpu VO_CPU.cpp )
target_include_directories(vocpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries( orb ${OpenCV_LIBS} )
target_link_libraries( vo ${OpenCV_LIBS} )
target_link_libraries( vocpu ${OpenCV_LIBS} )


