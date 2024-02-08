#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const double width = 1241.0;
const double height = 376.0;
const double fx = 718.8560;
const double fy = 718.8560;
const double cx = 607.1928;
const double cy = 185.2157;

int main() {
    string seq = "00";
    string gt_pose_dir = "/home/loahit/Downloads/Vis_Odo_project/poses/09.txt";
    string img_data_dir = "/home/loahit/Downloads/Vis_Odo_project/09/image_0";

    Mat trajMap = Mat::zeros(1000, 1000, CV_8UC3);

    vector<String> img_list;
    glob(img_data_dir + "/*.png", img_list, false);
    sort(img_list.begin(), img_list.end());
    int num_frames = img_list.size();
    cout << num_frames << endl;

    Mat prev_R = Mat::eye(3, 3, CV_64F);
    Mat prev_t = Mat::zeros(3, 1, CV_64F);

    cuda::GpuMat d_prev_img;
    cuda::GpuMat d_curr_img;
    cuda::GpuMat d_prev_descriptors, d_curr_descriptors;

    cuda::GpuMat d_matches_mask;
    Ptr<cuda::ORB> orb = cuda::ORB::create(6000);

    for (int i = 0; i < num_frames; ++i) {
        Mat curr_img = imread(img_list[i], IMREAD_GRAYSCALE);

        cuda::GpuMat d_curr_img(curr_img);

        Mat curr_R, curr_t;

        if (i == 0) {
            curr_R = Mat::eye(3, 3, CV_64F);
            curr_t = Mat::zeros(3, 1, CV_64F);
        } else {
            cuda::GpuMat d_prev_img(prev_img);

            cuda::GpuMat d_kp1, d_kp2;
            cuda::GpuMat d_des1, d_des2;

            orb->detectAndComputeAsync(d_prev_img, cuda::GpuMat(), d_kp1, d_des1);
            orb->detectAndComputeAsync(d_curr_img, cuda::GpuMat(), d_kp2, d_des2);

            cuda::BFMatcher_CUDA bf(NORM_HAMMING);
            vector<DMatch> matches;

            bf.match(d_des1, d_des2, matches);

            cuda::GpuMat d_pts1, d_pts2;
            cuda::drawKeypointsAsync(d_curr_img, d_kp2, d_curr_img, Scalar(0, 255, 0));

            cuda::GpuMat d_pts1_host, d_pts2_host;
            cuda::convertPointsFromHomogeneous(d_pts1, d_pts1_host);
            cuda::convertPointsFromHomogeneous(d_pts2, d_pts2_host);

            cuda::GpuMat d_E, d_mask;
            cuda::findEssentialMat(d_pts1, d_pts2, d_E, RANSAC, 0.999, 1.0, d_mask);

            cuda::GpuMat d_R, d_t;
            cuda::recoverPose(d_E, d_pts1, d_pts2, d_R, d_t);

            d_R.download(curr_R);
            d_t.download(curr_t);

            if (i == 1) {
                curr_R.copyTo(prev_R);
                curr_t.copyTo(prev_t);
            } else {
                curr_R = prev_R * curr_R;
                curr_t = prev_R * curr_t + prev_t;
            }

            int offset_draw = 1000 / 2;
            circle(trajMap, Point(curr_t.at<double>(0) + offset_draw, curr_t.at<double>(2) + offset_draw), 1, Scalar(255, 0, 0), 2);
            imshow("Trajectory", trajMap);
            waitKey(1);
        }

        prev_R = curr_R;
        prev_t = curr_t;

        img_list[i].erase(img_list[i].begin() + 50);
        img_list[i].erase(img_list[i].begin() + 50);
    }

    imwrite("trajMap.png", trajMap);

    return 0;
}
