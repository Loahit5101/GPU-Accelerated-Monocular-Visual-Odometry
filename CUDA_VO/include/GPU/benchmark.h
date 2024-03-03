#ifndef BENCHMARK__H  // Header guard to prevent multiple inclusions
#define BENCHMARK__H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "GPU/ORB.cuh"

#include "CPU/ORB_CPU.h"
using namespace std;
using namespace cv;

#endif // BENCHMARK_H