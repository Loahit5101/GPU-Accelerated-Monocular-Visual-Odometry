#include <opencv2/opencv.hpp>
#include <string>
#include <nmmintrin.h>
#include <chrono>

using namespace std;

void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors) {
  const int half_patch_size = 8;
  const int half_boundary = 16;
  int bad_points = 0;
  
  //remove boundary points
  for (auto &kp: keypoints) {
    if (kp.pt.x < half_boundary || kp.pt.y < half_boundary ||
        kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary) {
      // outside
      bad_points++;
      descriptors.push_back({});
      continue;
    }

  // compute moments
    float m01 = 0, m10 = 0;
    for (int dx = -half_patch_size; dx < half_patch_size; ++dx) {
      for (int dy = -half_patch_size; dy < half_patch_size; ++dy) {
        uchar pixel = img.at<uchar>(kp.pt.y + dy, kp.pt.x + dx);
        m10 += dx * pixel;
        m01 += dy * pixel;
      }
    }

    // angle should be arc tan(m01/m10);
    float m_sqrt = sqrt(m01 * m01 + m10 * m10) + 1e-18; // avoid divide by zero
    float sin_theta = m01 / m_sqrt;
    float cos_theta = m10 / m_sqrt;


}

int main(int argc, char **argv) {

  // load image
  cv::Mat first_image = cv::imread(first_file, 0);
  cv::Mat second_image = cv::imread(second_file, 0);
  assert(first_image.data != nullptr && second_image.data != nullptr);

  // detect FAST keypoints1 using threshold=40
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  vector<cv::KeyPoint> keypoints1;
  
  cv::FAST(first_image, keypoints1, 40); 
  vector<DescType> descriptor1;
  
  }
  
/*TODO
1. Complete ORB
2. Add own FAST implementation
3. implement BF matcher
*/
