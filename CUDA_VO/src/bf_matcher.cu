#include<stdio.h>
#include<stdlib.h>
#include <nmmintrin.h>
#include "ORB_CPU.h"

#define N 8
#define BLOCK_SIZE 32

__global__ void bfMatcherCUDA(int *a, int *b,int C_rows,int C_cols) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  if( row < C_rows && col < C_cols ){

      int z = a[row * C_cols + col];
      int x = b[row * C_cols + col];
      printf("%d = %d\n",z,x);
     
      int distance=0;
      unsigned char xor_result = (a[a[row * C_cols + col]]^ b[a[row * C_cols + col]]);

      while (xor_result){
          distance += xor_result & 1;
          xor_result >>= 1;
        }
      printf("distance is %d",distance);

  }
 
}

void fill_array(int *data, vector<DescType>& d) {
	for(int i=0;i<d.size();i++)
    {
      for(int j=0;j<d[i].size();j++){
      
   		data[i*d[i].size()+j] = d[i][j];
      }
    
   }
}


int main(void) {

  vector<DescType> desc1;
  vector<DescType> desc2;

  desc1.push_back({1,2,3,4});desc1.push_back({5,6,7,8});
  desc2.push_back({2,4,6,8});desc2.push_back({10,12,14,16});

  int *a, *b;
  int *d_a, *d_b;
  int a_size=0;
  int b_size=0;

  for(const auto& d:desc1)
  {
     a_size+=d.size();  

  } 
    for(const auto& d:desc2)
  {
     b_size+=d.size();  

  } 
   a_size = a_size* sizeof(int);
   b_size = a_size* sizeof(int);

	a = (int *)malloc(a_size); fill_array(a,desc1);
	b = (int *)malloc(b_size); fill_array(b,desc2);

  dim3 dim_grid(ceilf(desc1.size()/(float)BLOCK_SIZE), ceilf(desc1.size()/(float)BLOCK_SIZE), 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  cudaMalloc((void **)&d_a, a_size);
  cudaMalloc((void **)&d_b, b_size);
     

  cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);
 
  bfMatcherCUDA<<<dim_grid, dim_block>>>(d_a,d_b,desc1.size(),desc1[0].size());

  cudaDeviceSynchronize();

  cudaFree(a);
  cudaFree(b);

	return 0;
}