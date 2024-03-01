# GPU accelerated Monocular Visual Odometry

This project aims to build a Monocular Visual odometry pipeline using C++, CUDA and OpenCV on the KITTI odometry dataset.  

Main goal is to implement ORB features using CUDA and compare performance improvements. 

It implmenets ORB feature detection from scratch using CPU and CUDA.

## Usage

C++

```
mkdir build
cmake ..
make
./vo
```
For CPU VO:

```
./vocpu
```


For GPU VO:

```
./vogpu
```




https://github.com/Loahit5101/Monocular-Visual-Odometry/assets/55102632/4bd119fe-8dd6-41ba-b8f6-920d4204b839




TODO:

Compare and benchmark performance

Write headers and comments.Rewrite into Classes
