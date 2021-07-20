#pragma once
#include <iostream>
#include <opencv2/core/types.hpp>
#include "tkdnn.h"

namespace tk { namespace dnn {

cv::Mat vizFloat2colorMap(cv::Mat map, double min=0, double max=0, int classes=19);
cv::Mat vizData2Mat(dnnType *dataInput, tk::dnn::dataDim_t dim, int img_h, int img_w, double min=0, double max=0, int classes=19);
cv::Mat vizLayer2Mat(tk::dnn::Network *net, int layer, int imgdim = 1000);
    
}}
