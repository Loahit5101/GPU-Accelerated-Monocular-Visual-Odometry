#ifndef VO_CLASS__H  // Header guard to prevent multiple inclusions
#define VO_CLASS__H

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

class VisualOdometry{

public:

    string descriptor_type;
    
    VisualOdometry(const string& s): descriptor_type(s) {
       
        cout<<"hello";
        trajMap = Mat::zeros(1000, 1000, CV_8UC3);
        prev_R = Mat::eye(3, 3, CV_64F);
        prev_t = Mat::zeros(3, 1, CV_64F);
    }
    void run_VO(const vector<String>& img_list);


private:
    
    Mat trajMap;
    Mat prev_R, prev_t;
    vector<KeyPoint> prev_kp;
    vector<DescType> prev_descriptor;
    vector<KeyPoint> current_kp;
    vector<DescType> current_descriptor;
    vector<cv::DMatch> matches;
    vector<Point2f> pts1, pts2;
    

    void computeCameraPose(const vector<Point2f>& pts1, const vector<Point2f>& pts2, Mat& curr_R, Mat& curr_t);
    void computeKeypoints(const Mat& image, vector<KeyPoint>& keypoints);
    void computeDescriptor(const Mat& curr_img, vector<KeyPoint>& current_kp,vector<DescType>& current_descriptor);
    void matchFeatures(const vector<KeyPoint>& prev_kp, const vector<KeyPoint>& current_kp, const vector<DescType>& prev_descriptor, const vector<DescType>& current_descriptor,vector<cv::DMatch>& matches,vector<Point2f>& pts1,vector<Point2f>& pts2);
    
};

#endif // VO_CLASS__H