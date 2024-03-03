
#include "GPU/VO_class.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <img_data_dir> <descriptor_type>" << endl;
        return 1;
    }

    string img_data_dir = argv[1];
    string descriptor_type = argv[2];

    vector<String> img_list;
    glob(img_data_dir + "/*.png", img_list, false);
    sort(img_list.begin(), img_list.end());
    int num_frames = img_list.size();

    for (int i = 0; i < num_frames; ++i) {
        img_list[i].erase(img_list[i].begin() + 50); // Assuming you want to erase characters at index 50 twice
        img_list[i].erase(img_list[i].begin() + 50);
    }

    VisualOdometry VO(descriptor_type);
    VO.run_VO(img_list);

    return 0;
}