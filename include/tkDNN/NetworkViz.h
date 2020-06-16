#pragma once
#include <iostream>
#include <opencv2/core/types.hpp>
#include "tkdnn.h"

namespace tk { namespace dnn {

cv::Mat vizFloat2colorMap(cv::Mat map);
cv::Mat vizData2Mat(dnnType *dataInput, tk::dnn::dataDim_t dim, int imgdim);
cv::Mat vizLayer2Mat(tk::dnn::Network *net, int layer, int imgdim = 1000);
    
}}
