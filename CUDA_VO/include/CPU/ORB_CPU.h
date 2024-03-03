#include <opencv2/opencv.hpp>
#include <string>
#include <nmmintrin.h>
#include <chrono>

using namespace std;
typedef vector<uint32_t> DescType;
vector<DescType> ComputeORB_CPU(const cv::Mat &img,const vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);
void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches);


