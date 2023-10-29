#include<stdio.h>
#include<stdlib.h>
#include <nmmintrin.h>
#include "ORB_CPU.h"

#define N 8

__global__ void loop(int *a, int *b) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
 
  int distance = (a[index]^ b[index]);

  unsigned char xor_result = a[index]^ b[index];
     
  while (xor_result){
          distance += xor_result & 1;
          xor_result >>= 1;
        }
  printf("distance is %d",distance);


}

void fill_array(int *data, DescType& d) {
	for(int idx=0;idx<d.size();idx++)
		data[idx] = d[idx];
}


int main(void) {


  vector<DescType> desc1;
  vector<DescType> desc2;

  desc1.push_back({1,2,3,4});
  desc2.push_back({1,2,3,4});
 

  int *a, *b;
  int *d_a, *d_b; 
  int size = desc1[0].size()* sizeof(int);


	a = (int *)malloc(size); fill_array(a,desc1[0]);
	b = (int *)malloc(size); fill_array(b,desc2[0]);


  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
     

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  int threads_per_block = 4;
	int no_of_blocks = N/threads_per_block;	
	loop<<<no_of_blocks,threads_per_block>>>(d_a,d_b);


  cudaDeviceSynchronize();


	return 0;
}