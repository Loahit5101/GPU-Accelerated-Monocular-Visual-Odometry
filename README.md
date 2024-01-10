# GPU accelerated Monocular Visual Odometry

This project aims to build a Monocular Visual odometry pipeline using C++, CUDA and OpenCV on the KITTI odometry dataset.  

Main goal is to implement FAST keypoints and ORB features using CUDA and compare performance improvements. 

Currently it uses openCV to detect ORB features. Implementation of ORB feature detection from scratch using CUDA is under progress.

## Usage

C++

```
mkdir build
cmake ..
make
./vo
```





https://github.com/Loahit5101/Monocular-Visual-Odometry/assets/55102632/4bd119fe-8dd6-41ba-b8f6-920d4204b839




TODO:


Implement FAST keypoint detection using CUDA - done -> TODO: integrate with VO

Implement ORB from scratch using CPU (done) and GPU

Compare and benchmark performance

Write headers and comments.Rewrite into Classes
