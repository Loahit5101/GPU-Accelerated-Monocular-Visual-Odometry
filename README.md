# GPU accelerated Monocular Visual Odometry

1. A Monocular Visual odometry pipeline using Modern C++, CUDA and OpenCV on the KITTI odometry dataset.
   
2. ORB feature computation is implemented using **CUDA kernels** and compares performance improvements. CPU or GPU version can be chosen using input arguments.  

      ![cudaVO-2024-03-03_21 53 14-ezgif com-crop (1)](https://github.com/Loahit5101/GPU-Accelerated-Monocular-Visual-Odometry/assets/55102632/1bb1497a-0201-4caf-a657-ac0a1fad4f9e)

## Pipeline Overview:
1. **FAST Keypoint Detection**: Detect keypoints in the images using the FAST algorithm.
2. **ORB Descriptor Computation**: Compute ORB descriptors for the detected keypoints.
3. **Feature Matching**: Match keypoints between images using a matching algorithm.
4. **Essential Matrix Estimation using RANSAC**: Estimate the Essential Matrix (`E`) using RANSAC to handle outliers.
5. **Compute Pose from E**: Extract camera pose information from the estimated Essential Matrix (`E`).
6. **Pose Tracking**: Track the camera pose over time using the computed poses.

## Dependencies
- OpenCV 4.2.0
- CUDA
  
## Usage

```
mkdir build
cmake -DCMAKE_CUDA_ARCHITECTURES= $your_architecture ..
make

```

- **CPU_VO**: Run CPU Visual Odometry.
    ```bash
    ./main IMAGE_PATH ORB_CPU
    ```

- **GPU_VO**: Run CUDA-accelerated Visual Odometry.
    ```bash
    ./main IMAGE_PATH ORB_GPU
    ```

- **benchmark**: Run benchmarking to evaluate the performance of the GPU-accelerated implementation.
    ```bash
    ./benchmark
    ```
## Result

## Computation Time Comparison

| Method   | Compute Time for 1600 Images (ms) |
|----------|-----------------------------------|
| ORB_CPU  | 20134 ms                          |
| ORB_CUDA | 4474 ms                           |


TODO:

Add documentation in code

Add tests
