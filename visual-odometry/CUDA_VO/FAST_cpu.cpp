#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#define CIRCLE_SIZE 16
#define MASK_SIZE 3
#define PADDING 3

void show_image(cv::Mat img) {
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", img);
	cv::waitKey(0);
}
typedef struct corner {
	unsigned score;
	unsigned x;
	unsigned y;

} corner;

bool comparator(unsigned char pixel_val, unsigned char circle_val, int threshold, char sign) {
	/// return boolean if true ... sign parameter gives us criterion
	if (sign == 1) {
		return circle_val > (pixel_val + threshold);
	}
	else {
		return circle_val < (pixel_val - threshold);
	}
}

int fast_test(unsigned char *input, int *circle, int threshold, int id) {
	unsigned char pixel = input[id];
	unsigned char top = input[id + circle[0]];
	unsigned char right = input[id + circle[4]];
	unsigned char down = input[id + circle[8]];
	unsigned char left = input[id + circle[12]];

	unsigned char sum = comparator(pixel, top, threshold, 1) + comparator(pixel, right, threshold, 1) +
						comparator(pixel, down, threshold, 1) + comparator(pixel, left, threshold, 1);
	if (sum < 3) {
		sum = comparator(pixel, top, threshold, -1) + comparator(pixel, right, threshold, -1) +
			  comparator(pixel, down, threshold, -1) + comparator(pixel, left, threshold, -1);
		if (sum < 3) {
			return 1;
		}
	}
	return 0;
}

int get_score(int pixel_val, int circle_val, int threshold) {
	/// returns score of circle element, positive when higher, negative when lower intensity
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
int complex_test(unsigned char *input, unsigned *scores, unsigned *corner_bools, int *circle, int threshold, int pi, int s_id, int g_id) {	
	/// make complex test and calculate score
	unsigned char pixel = input[s_id];
	int score;
	int score_sum = 0;
	int max_score = 0;
	char val;
	char last_val = -2;
	unsigned char consecutive = 1;
	bool corner = false;
	/// iterate over whole circle
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


int threshold=75;

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
	// create mask with given defined mask size and width
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

std::vector<corner> cpu_FAST(unsigned char *input, unsigned *scores, int *mask, int *circle, int width, int height) {
	/// fast test
	std::vector<corner> ret;
	int id1d;
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			scores[id1d] = fast_test(input, circle, threshold, id1d);
		}
	}
	/// complex test
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			if (scores[id1d] > 0) {
				scores[id1d] = complex_test(input, scores, scores, circle, threshold, 3.14, id1d, id1d);
			}
		}
	}
	/// non-max suppression
	bool is_max;
	int val;
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			val = scores[id1d];
			if (val > 0) {
				is_max = true;
				for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
				{
					if (val < scores[id1d + mask[i]]) {
						is_max = false;
						break;
					}
				}
				if (is_max) {
					corner c;
					c.score = (unsigned)val;
					c.x = (unsigned)x;
					c.y = (unsigned)y;
					ret.push_back(c);
				}
			}
		}
	}
	return ret;
}


void run_on_cpu(cv::Mat image) {
	
		std::vector<cv::KeyPoint> keypointsD;

		cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, true);
		detector->detect(image, keypointsD, cv::Mat());
		// cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < keypointsD.size(); i++) {
			cv::circle(image, keypointsD[i].pt, 5, cv::Scalar(0, 255, 0), 2);
		}
	
	
	//cv::Size size(1280, 720);	// resize for testing
	//resize(image, image, size);
	//show_image(image);
}



int main(int argc, char **argv)
{

	/// load image

       cv::Mat image;
     
       image = cv::imread("image.jpg", 1);

	run_on_cpu(image);
        cv::imwrite("output.jpg", image);
			
		
	}
