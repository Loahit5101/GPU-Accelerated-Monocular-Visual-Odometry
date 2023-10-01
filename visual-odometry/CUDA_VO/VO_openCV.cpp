#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
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

        for (int i = 0; i < num_frames; ++i) {
       
        img_list[i].erase(img_list[i].begin() + 50);
        img_list[i].erase(img_list[i].begin() + 50);
        
        }
    Mat prev_R = Mat::eye(3, 3, CV_64F);
    Mat prev_t = Mat::zeros(3, 1, CV_64F);

    for (int i = 0; i < num_frames; ++i) {
        Mat curr_img = imread(img_list[i], IMREAD_GRAYSCALE);

        Mat curr_R, curr_t;

        if (i == 0) {
            curr_R = Mat::eye(3, 3, CV_64F);
            curr_t = Mat::zeros(3, 1, CV_64F);
        } else {
            Mat prev_img = imread(img_list[i - 1], IMREAD_GRAYSCALE);

            Ptr<ORB> orb = ORB::create(6000);
            vector<KeyPoint> kp1, kp2;
            Mat des1, des2;
            orb->detectAndCompute(prev_img, Mat(), kp1, des1);
            orb->detectAndCompute(curr_img, Mat(), kp2, des2);

            BFMatcher bf(NORM_HAMMING, true);
            vector<DMatch> matches;
            bf.match(des1, des2, matches);
            sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) {
                return a.distance < b.distance;
            });

            //Mat img_matching;
            //drawMatches(prev_img, kp1, curr_img, kp2, matches, img_matching);
            //imshow("feature matching", img_matching);

            vector<Point2f> pts1, pts2;
            for (const DMatch &match : matches) {
                pts1.push_back(kp1[match.queryIdx].pt);
                pts2.push_back(kp2[match.trainIdx].pt);
            }

            Mat E, mask;
            E = findEssentialMat(pts1, pts2, fx, Point2d(cx, cy), RANSAC, 0.999, 1.0, mask);
            recoverPose(E, pts1, pts2, curr_R, curr_t, fx, Point2d(cx, cy), mask);

            if (i == 1) {
                curr_R.copyTo(prev_R);
                curr_t.copyTo(prev_t);
            } else {
                curr_R = prev_R * curr_R;
                curr_t = prev_R * curr_t + prev_t;
            }

            Mat curr_img_kp;
            drawKeypoints(curr_img, kp2, curr_img_kp, Scalar(0, 255, 0));
            imshow("keypoints from current image", curr_img_kp);
        }
        
        prev_R = curr_R;
        prev_t = curr_t;
        cout<<"curr_t.at<double>(0)"<<curr_t.at<double>(0)<<'\n';
        cout<<"curr_t.at<double>(2)"<<curr_t.at<double>(2)<<'\n';
   
        int offset_draw = 1000 / 2;
        circle(trajMap, Point(curr_t.at<double>(0) + offset_draw, curr_t.at<double>(2) + offset_draw), 1, Scalar(255, 0, 0), 2);
        imshow("Trajectory", trajMap);
        waitKey(1);
    }

    imwrite("trajMap.png", trajMap);

    return 0;
}
