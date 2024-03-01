#ifndef FAST_TEST_H
#define FAST_TEST_H

#include "cuda.cuh"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>

int threshold = 75;
int mode = 1;
int pi = 12;
char *filename = NULL;
int circle_size = 5;

/// host stuff
unsigned char *h_img;
unsigned *h_corner_bools;
int *h_circle;
int *h_mask;
int *h_mask_shared;

cudaStream_t memory_s, work_s;

clock_t start, end;
double time_measured;

#endif