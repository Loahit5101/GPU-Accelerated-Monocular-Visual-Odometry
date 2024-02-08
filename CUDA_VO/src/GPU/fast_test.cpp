# include "fast_test.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>


void create_circle(int *circle, int w) {
	circle[0] = -3 * w;
	circle[1] = -3 * w + 1;
	circle[2] = -2 * w + 2;
	circle[3] = -w + 3;

	circle[4] = 3;
	circle[5] = w + 3;
	circle[6] = 2 * w + 2;
	circle[7] = 3 * w + 1;

	circle[8] = 3 * w;
	circle[9] = 3 * w - 1;
	circle[10] = 2 * w - 2;
	circle[11] = w - 3;

	circle[12] = -3;
	circle[13] = -w - 3;
	circle[14] = -2 * w - 2;
	circle[15] = -3 * w - 1;
}

void create_mask(int *mask, int w) {
	
	int start = -(int)MASK_SIZE / 2;
	int end = (int)MASK_SIZE / 2;
	int index = 0;
	for (int i = start; i <= end; i++)
	{
		for (int j = start; j <= end; j++)
		{
			mask[index] = i * w + j;
			index++;
		}
	}
}

void fill_gpu_const_mem(int width, int shared_width) {
	/// create circle and mask and copy to device
	if (mode == 3) {
		printf("--- Using shared memory --- \n");
		create_circle(h_circle, shared_width);
		create_mask(h_mask_shared, shared_width);
		create_mask(h_mask, width);
		fill_const_mem(h_circle, h_mask, h_mask_shared);
	}
	else {
		printf("--- Using global memory --- \n");
		create_circle(h_circle, width);
		create_mask(h_mask_shared, shared_width);
		create_mask(h_mask, width);
		fill_const_mem(h_circle, h_mask, h_mask_shared);
	}
}


void init_gpu(cv::Mat image, int length, int shared_width) {
	size_t char_size = length * sizeof(unsigned char);
	size_t int_size = length * sizeof(unsigned int);
	printf("--- GPU memory initialized --- \n");

	/// allocate memory
	h_img = (unsigned char*)malloc(char_size);
	h_corner_bools = (unsigned*)malloc(int_size);
	h_circle = (int*)malloc(CIRCLE_SIZE * sizeof(int));
	h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE * sizeof(int));
	h_mask_shared = (int*)malloc(MASK_SIZE*MASK_SIZE * sizeof(int));
	CHECK_ERROR(cudaMalloc((void**)&d_img_old, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_img_new, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_corner_bools, int_size));
	CHECK_ERROR(cudaMalloc((void**)&d_scores, int_size));

	fill_gpu_const_mem(image.cols, shared_width);
}


int main(){

    cv::Mat image;
	cv::Mat image_gray;
	image = cv::imread("/home/loahit/Downloads/Vis_Odo_project/09/image_0/000000.png", 1);
    //cv::imshow("Display window", image);
	//cv::waitKey(0);

    int length = image.cols*image.rows;
    int shared_width = BLOCK_SIZE + (2 * PADDING);
	init_gpu(image, length, shared_width);
	int number_of_corners;
	cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
     

    return 0;
}