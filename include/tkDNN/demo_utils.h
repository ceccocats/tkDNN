#ifndef DEMO_UTILS_H
#define DEMO_UTILS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include "utils.h"

#ifdef __linux__
#include <unistd.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <yaml-cpp/yaml.h>


TKDNN_LIB_EXPORT_API void readCalibrationMatrix(const std::string& path, cv::Mat& calib_mat);

#endif //DEMO_UTILS_H