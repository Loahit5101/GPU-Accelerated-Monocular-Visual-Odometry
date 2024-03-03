#include "GPU/benchmark.h"
#include "CPU/ORB_CPU.h"

const double width = 1241.0;
const double height = 376.0;
const double fx = 718.8560;
const double fy = 718.8560;
const double cx = 607.1928;
const double cy = 185.2157;

double cpu_time = 0.0;
double gpu_time = 0.0;
using namespace std::chrono;

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
    cout << img_list[num_frames]<< endl;

    for (int i = 0; i < num_frames; ++i) {
       
        img_list[i].erase(img_list[i].begin() + 50);
        img_list[i].erase(img_list[i].begin() + 50);
        
    }
        
    Mat prev_R = Mat::eye(3, 3, CV_64F);
    Mat prev_t = Mat::zeros(3, 1, CV_64F);

    int j=0;
    while(j<2)
    {
    for (int i = 0; i < num_frames/2; ++i) {

        cout<<img_list[i]<<endl;
        Mat curr_img = imread(img_list[i], IMREAD_GRAYSCALE);


        Mat curr_R, curr_t;

        if (i == 0) {
            curr_R = Mat::eye(3, 3, CV_64F);
            curr_t = Mat::zeros(3, 1, CV_64F);
        } 
        
        else {

            Mat prev_img = imread(img_list[i - 1], IMREAD_GRAYSCALE);

            vector<KeyPoint> kp1, kp2;
            
            
            cv::FAST(prev_img, kp1, 40);

            cv::FAST(curr_img, kp2, 40);

            std::cout<<"KP SIZE =="<<kp2.size()<<std::endl;

            if(j==0)
            {
            vector<DescType> des1;
            vector<DescType> des2;
            
            auto start_CPU = high_resolution_clock::now(); 
            des1 = ComputeORB_CPU(prev_img, kp1, des1);
            des2 =ComputeORB_CPU(curr_img, kp2, des2);
             auto stop_CPU = high_resolution_clock::now(); // Record stop time
            auto duration_CPU = duration_cast<milliseconds>(stop_CPU - start_CPU);
            cout << "Time taken by CPU: " << duration_CPU.count() << " milliseconds" << endl;
            cpu_time += duration_CPU.count();

            }

            
            if(j==1)
            {
            vector<DescType> des3;
            vector<DescType> des4;
            auto start_GPU = high_resolution_clock::now(); 
            des4 = ComputeORB_CUDA(curr_img,kp2,des4);
            des3 = ComputeORB_CUDA(prev_img,kp1,des3);
            auto stop_GPU = high_resolution_clock::now(); // Record stop time
            auto duration_GPU = duration_cast<milliseconds>(stop_GPU - start_GPU);
            cout << "Time taken by GPU: " << duration_GPU.count() << " milliseconds" << endl;
            
            gpu_time += duration_GPU.count();
            }

            
        }
        
    }
    j+=1;
    }


    cout<<" Total CPU time = "<<cpu_time<<std::endl;
    cout<<" Total GPU time = "<<gpu_time<<std::endl;

    cout<<" GPU ORB is "<<round((cpu_time/gpu_time) * 100) / 100.0<<" times faster than CPU ORB!";

    
    return 0;
}
