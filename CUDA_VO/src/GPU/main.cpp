
#include "GPU/VO_class.h"

int main(){

    string img_data_dir = "/home/loahit/Downloads/Vis_Odo_project/09/image_0";
    vector<String> img_list;
    glob(img_data_dir + "/*.png", img_list, false);
    sort(img_list.begin(), img_list.end());
    int num_frames = img_list.size();

    for (int i = 0; i < num_frames; ++i) { img_list[i].erase(img_list[i].begin() + 50); img_list[i].erase(img_list[i].begin() + 50);}

    string s = "ORB_CPU";

    VisualOdometry VO(s);
    VO.run_VO(img_list);
    
    return 0;
}