#include<stdio.h>
#include<stdlib.h>

#include "ORB_CPU.h"
#include <cuda_runtime.h>

__global__ void BfMatchKernel(const DescType* desc1, const DescType* desc2, cv::DMatch* matches, int d_max, int desc1Size, int desc2Size) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  


}

void BfMatch_CUDA(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches) {
  const int d_max = 40;
  int desc1Size = desc1.size();
  int desc2Size = desc2.size();

  // Allocate device memory
  DescType* d_desc1;
  DescType* d_desc2;
  cv::DMatch* d_matches;

  cudaMalloc((void**)&d_desc1, desc1Size * sizeof(DescType));
  cudaMalloc((void**)&d_desc2, desc2Size * sizeof(DescType));
  cudaMalloc((void**)&d_matches, desc1Size * sizeof(cv::DMatch));

  // Copy data from host to device
  cudaMemcpy(d_desc1, desc1.data(), desc1Size * sizeof(DescType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_desc2, desc2.data(), desc2Size * sizeof(DescType), cudaMemcpyHostToDevice);

  // Launch the CUDA kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (desc1Size + threadsPerBlock - 1) / threadsPerBlock;

  BfMatchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_desc1, d_desc2, d_matches, d_max, desc1Size, desc2Size);

  // Copy results back to the host
  cudaMemcpy(matches.data(), d_matches, desc1Size * sizeof(cv::DMatch), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_desc1);
  cudaFree(d_desc2);
  cudaFree(d_matches);
}
int main()
{

	return 0;
}