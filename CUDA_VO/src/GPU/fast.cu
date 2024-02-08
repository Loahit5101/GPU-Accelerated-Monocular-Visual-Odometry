#include "cuda.cuh"

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
__device__ __host__  int get_score(int pixel_val, int circle_val, int threshold) {
	
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

__host__ void fill_const_mem(int *h_circle, int *h_mask, int *h_mask_shared) {
	CHECK_ERROR(cudaMemcpyToSymbol(d_circle, h_circle, CIRCLE_SIZE * sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_mask, h_mask, MASK_SIZE * MASK_SIZE * sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_mask_shared, h_mask_shared, MASK_SIZE * MASK_SIZE * sizeof(int)));
	CHECK_ERROR(cudaDeviceSynchronize());
	return;
}

__device__ __host__ int fast_test(unsigned char *input, unsigned *scores, unsigned *corner_bools, int *circle, int threshold, int pi, int s_id, int g_id) {	
	
	unsigned char pixel = input[s_id];
	int score;
	int score_sum = 0;
	int max_score = 0;
	char val;
	char last_val = -2;
	unsigned char consecutive = 1;
	bool corner = false;

	for (size_t i = 0; i < (CIRCLE_SIZE + pi); i++) 
	{
		if (consecutive >= pi) {
			corner = true;
			if (score_sum > max_score) {
				max_score = score_sum;
			}
		}
		score = get_score(pixel, input[s_id + circle[i % CIRCLE_SIZE]], threshold);
		/// signum
		val = (score < 0) ? -1 : (score > 0); 
		if (val != 0 && val == last_val) {
			consecutive++;
			score_sum += abs(score);
		}
		else {
			consecutive = 1;
			score_sum = abs(score);
		}
		last_val = val;
	}
	if (corner) {
		if (score_sum > max_score) {
			max_score = score_sum;
		}
		corner_bools[g_id] = 1;
		scores[g_id] = max_score;
		return max_score;
	}
	else {
		return 0;
	}
}

__global__ void FAST(unsigned char *input, unsigned *scores, unsigned *corner_bools, int width, int height, int threshold, int pi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	/// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	if (id1d == -1) {
		return;
	}

	int max_score = fast_test(input, scores, corner_bools, d_circle, threshold, pi, id1d, id1d);
	
	__syncthreads();

	bool erase = false;
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
	{
		if (scores[id1d + d_mask[i]] > max_score) {
			erase = true;
			break;
		}
	}
	__syncthreads();
	if (erase) {
		scores[id1d] = 0;
		corner_bools[id1d] = 0;
	}
	return;
}

