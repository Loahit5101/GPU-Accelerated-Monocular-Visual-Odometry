#include "cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


__device__  char comparator(unsigned char pixel_val, unsigned char circle_val, int threshold, char sign) {
	/// return boolean if true ... sign parameter gives us criterion
	if (sign == 1) {
		return circle_val > (pixel_val + threshold);
	}
	else {
		return circle_val < (pixel_val - threshold);
	}
}

//Calculate element of score of given pixel
__device__  int get_score(int pixel_val, int circle_val, int threshold) {
	
	int val = pixel_val + threshold;
	if (circle_val > val) {
		return circle_val - val;
	}
	else {
		val = pixel_val - threshold;
		if (circle_val < val) {
			return circle_val - val;
		}
		else {
			return 0;
		}
	}
}

// Recalculate 2D indexing into 1D
 
__device__ int coords_2to1(int x, int y, int width, int height, bool eliminate_padding) {
	if (eliminate_padding && ((x - PADDING) < 0 || (x + PADDING) >= width || (y - PADDING) < 0 || (y + PADDING) >= height)) {
		/// cutout the borders of image, only active when eliminate_padding == true
		return -1;
	}
	else {
		return x + y * width;
	}
}

/__device__ int coords_2to1(int x, int y, int width, int height, bool eliminate_padding) {
	if (eliminate_padding && ((x - PADDING) < 0 || (x + PADDING) >= width || (y - PADDING) < 0 || (y + PADDING) >= height)) {
		
		return -1;
	}
	else {
		return x + y * width;
	}
}

	}
}
