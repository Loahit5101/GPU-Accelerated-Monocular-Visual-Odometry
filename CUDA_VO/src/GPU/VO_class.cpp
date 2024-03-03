
#include "GPU/VO_class.h"

const double width = 1241.0;
const double height = 376.0;
const double fx = 718.8560;
const double fy = 718.8560;
const double cx = 607.1928;
const double cy = 185.2157;


void VisualOdometry::computeCameraPose(const vector<Point2f>& pts1, const vector<Point2f>& pts2, Mat& curr_R,Mat& curr_t){

            Mat E, mask;
            E = findEssentialMat(pts1, pts2, fx, Point2d(cx, cy), RANSAC, 0.999, 1.0, mask);
            recoverPose(E, pts1, pts2, curr_R, curr_t, fx, Point2d(cx, cy), mask);
     
}

void VisualOdometry::computeKeypoints(const Mat& curr_img,vector<KeyPoint>& current_kp){

            cv::FAST(curr_img, current_kp, 40);
}

void VisualOdometry::matchFeatures(const vector<KeyPoint>& prev_kp, const vector<KeyPoint>& current_kp, const vector<DescType>& prev_descriptor, const vector<DescType>& current_descriptor,vector<cv::DMatch>& matches,vector<Point2f>& pts1,vector<Point2f>& pts2){

            BfMatch(prev_descriptor, current_descriptor, matches);

            sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) {
                return a.distance < b.distance;
            });
            
            for (const DMatch &match : matches) {
                pts1.push_back(prev_kp[match.queryIdx].pt);
                pts2.push_back(current_kp[match.trainIdx].pt);
            }
}
void VisualOdometry::computeDescriptor(const Mat& curr_img, vector<KeyPoint>& current_kp,vector<DescType>& current_descriptor){
     
     if(descriptor_type=="ORB_GPU"){
            current_descriptor = ComputeORB_CUDA(curr_img,current_kp,current_descriptor);}

     if(descriptor_type=="ORB_CPU"){

            current_descriptor = ComputeORB_CPU(curr_img,current_kp,current_descriptor);}       

}
void VisualOdometry::run_VO(const vector<String>& img_list){

  Mat prev_img = imread(img_list[0], IMREAD_GRAYSCALE);

  imshow("Trajectory", trajMap);
  int num_frames = img_list.size();
  cv::FAST(prev_img, prev_kp, 40);
  //imshow("Trajectory", prev_img);
  //waitKey(0);
  cout<<"HERE";

  if(descriptor_type=="ORB_GPU"){
            prev_descriptor = ComputeORB_CUDA(prev_img,prev_kp,prev_descriptor);}

  if(descriptor_type=="ORB_CPU"){

            prev_descriptor = ComputeORB_CPU(prev_img,prev_kp,prev_descriptor);}  
  
  for (int i = 0; i < num_frames/2; ++i) {

        Mat curr_img = imread(img_list[i], IMREAD_GRAYSCALE);
        Mat curr_R, curr_t;

        if (i == 0) {
            curr_R = Mat::eye(3, 3, CV_64F);
            curr_t = Mat::zeros(3, 1, CV_64F);
        } 
        
        else {

            computeKeypoints(curr_img,current_kp);

            computeDescriptor(curr_img,current_kp,current_descriptor);
 
            matchFeatures(prev_kp,current_kp,prev_descriptor,current_descriptor,matches,pts1,pts2);

            computeCameraPose(pts1, pts2, curr_R, curr_t);

            if (i == 1) {
                curr_R.copyTo(prev_R);
                curr_t.copyTo(prev_t);
            } else {
                curr_R = prev_R * curr_R;
                curr_t = prev_R * curr_t + prev_t;
            }

            Mat curr_img_kp;
            drawKeypoints(curr_img, current_kp, curr_img_kp, Scalar(0, 255, 0));
            imshow("keypoints from current image", curr_img_kp);

        // Avoid copy    
        prev_kp = std::move(current_kp);
        prev_descriptor = std::move(current_descriptor);

        matches.clear();
        pts1.clear();
        pts2.clear();
       
        }
        
        prev_img = curr_img;
        prev_R = curr_R;
        prev_t = curr_t;
        cout<<"curr_t.at<double>(0)"<<curr_t.at<double>(0)<<'\n';
        cout<<"curr_t.at<double>(2)"<<curr_t.at<double>(2)<<'\n';
   
        int offset_draw = 1000 / 2;
        circle(trajMap, Point(curr_t.at<double>(0) + offset_draw, curr_t.at<double>(2) + offset_draw), 1, Scalar(255, 0, 0), 2);
        imshow("Trajectory", trajMap);
        waitKey(1);
    }

}

int main(){

    string img_data_dir = "/home/loahit/Downloads/Vis_Odo_project/09/image_0";
    vector<String> img_list;
    glob(img_data_dir + "/*.png", img_list, false);
    sort(img_list.begin(), img_list.end());
    int num_frames = img_list.size();

    for (int i = 0; i < num_frames; ++i) { img_list[i].erase(img_list[i].begin() + 50); img_list[i].erase(img_list[i].begin() + 50);}

    string s = "ORB_GPU";

    VisualOdometry VO(s);
    VO.run_VO(img_list);

    return 0;
}
