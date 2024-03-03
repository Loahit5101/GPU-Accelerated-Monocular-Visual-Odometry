# GPU accelerated Monocular Visual Odometry

1. A Monocular Visual odometry pipeline using Modern C++, CUDA and OpenCV on the KITTI odometry dataset.  

2. ORB feature computation is implemented using CUDA and compares performance improvements. 

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

https://github.com/Loahit5101/Monocular-Visual-Odometry/assets/55102632/4bd119fe-8dd6-41ba-b8f6-920d4204b839



TODO:

Add documentation in code
Add tests
