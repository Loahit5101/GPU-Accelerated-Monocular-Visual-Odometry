#ifndef ORB_CU_H  
#define ORB_CU_H

#include <vector>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "CPU/ORB_CPU.h"

vector<DescType> getDescriptors(const uint32_t* descriptors_array, size_t num_descriptors);
__global__ void ComputeORBKernel(const uchar* img_data, int img_cols, int img_rows, int half_patch_size, int half_boundary, int num_keypoints, const cv::KeyPoint* keypoints, uint32_t* descriptors);
vector<DescType> ComputeORB_CUDA(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);

#endif // ORB_CU_H