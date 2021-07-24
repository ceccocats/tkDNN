
#include "demo_utils.h"

void readCalibrationMatrix(const std::string& path, cv::Mat& calib_mat){
    YAML::Node config   = YAML::LoadFile(path);

    //read camera matrix
    int rows = config["camera_matrix"]["rows"].as<int>();
    int cols = config["camera_matrix"]["cols"].as<int>();

    cv::Mat calib = cv::Mat(cv::Size(rows, cols), CV_32F);
    float *vals = (float *)calib.data;
    
    for(int i=0; i < config["camera_matrix"]["data"].size(); ++i )
        vals[i] = config["camera_matrix"]["data"][i].as<float>();

    calib_mat = calib;

}
