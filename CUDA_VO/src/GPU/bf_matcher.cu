#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "GPU/ORB.cuh"

using namespace std;
#define DESC_SIZE 8

__global__ void computeDistances(const uint32_t* desc1_gpu, const uint32_t* desc2_gpu, int* distances, int* match_idx, const int numDesc1, const int numDesc2, const int d_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numDesc1) {
        for (int i2 = 0; i2 < numDesc2; ++i2) {

            //printf("tid = %d, i2 = %d \n",tid,i2);
            int distance = 0;
            for (int k = 0; k < DESC_SIZE; ++k) {
                uint32_t xor_result = desc1_gpu[tid * DESC_SIZE + k] ^ desc2_gpu[i2 * DESC_SIZE + k];
                                
                // Count the number of set bits in xor_result
                while (xor_result) {
                    distance += xor_result & 1;
                    xor_result >>= 1;
                }
            }
            if (distance < d_max && distance < distances[tid]) {
                distances[tid] = distance;

                printf("%d \n",distances);
                match_idx[tid] = i2;
            }
        }

        printf("%d \n",distances[tid]);
    }
}

void bfMatch_CUDA(vector<DescType>& desc1, vector<DescType>& desc2, vector<cv::DMatch>& matches) {
  
    uint32_t* desc1_gpu;
    uint32_t* desc2_gpu;
    
    const int d_max = 40;
    const int numDesc1 = desc1.size();
    const int numDesc2 = desc2.size();

    cudaMalloc((void**)&desc1_gpu, numDesc1 * DESC_SIZE * sizeof(uint32_t));
    cudaMemcpy(desc1_gpu, desc1.data(), numDesc1 * DESC_SIZE * sizeof(DescType), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&desc2_gpu, numDesc2 * DESC_SIZE * sizeof(uint32_t));
    cudaMemcpy(desc2_gpu, desc2.data(), numDesc2 * DESC_SIZE * sizeof(DescType), cudaMemcpyHostToDevice);

    int* d_distances;
    cudaMalloc((void**)&d_distances, numDesc1 * sizeof(int));
    cudaMemset(d_distances, 256, numDesc1 * sizeof(int));

    int* d_match_idx;
    cudaMalloc((void**)&d_match_idx, numDesc1 * sizeof(int));
    cudaMemset(d_match_idx, 256, numDesc1 * sizeof(int));

    int block_size = 256;
    int num_blocks = (desc1.size() + block_size - 1) / block_size;

    computeDistances<<<num_blocks, block_size>>>(desc1_gpu, desc2_gpu, d_distances, d_match_idx, numDesc1, numDesc2, d_max);

    int* h_distances = new int[numDesc1];
    cudaMemcpy(h_distances, d_distances, numDesc1 * sizeof(int), cudaMemcpyDeviceToHost);

    int* h_match_idx = new int[numDesc1];
    cudaMemcpy(h_match_idx, d_match_idx, numDesc1 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numDesc1; ++i) {
          if (h_distances[i] < d_max) {
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = h_match_idx[i];
        match.distance = h_distances[i];
        matches.push_back(match);

        //std::cout<<h_distances[i]<<std::endl;
    }
    }

    delete[] h_distances;

}

int main() {
    // Example usage
    vector<DescType> desc1, desc2;
    vector<cv::DMatch> matches;

    cv::Mat image1;
    cv::Mat image2;

    image1 = cv::imread("/home/loahit/Downloads/Vis_Odo_project/09/image_0/000000.png", cv::IMREAD_GRAYSCALE);
    image2 = cv::imread("/home/loahit/Downloads/Vis_Odo_project/09/image_0/000001.png", cv::IMREAD_GRAYSCALE);

    vector<cv::KeyPoint> kp1;
    vector<cv::KeyPoint> kp2;

    cv::FAST(image1, kp1, 40);
    cv::FAST(image2, kp2, 40);

    desc1 = ComputeORB_CUDA(image1,kp1,desc1);
    desc2 = ComputeORB_CUDA(image2,kp2,desc2);

    bfMatch_CUDA(desc1,desc2,matches);

    vector<cv::Point2f> pts1;vector<cv::Point2f> pts2;

            sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) {
                return a.distance < b.distance;
            });
            
            for (const cv::DMatch &match : matches) {
                pts1.push_back(kp1[match.queryIdx].pt);
                pts2.push_back(kp2[match.trainIdx].pt);
            }

    // Draw matches
    cv::Mat img_matches;
    cv::drawMatches(image1, kp1, image2, kp2, matches, img_matches);

    // Show or save the image with matches
    cv::imshow("Matches", img_matches);
    cv::waitKey(0); // Wait for a key press to close the window

    return 0;
}
