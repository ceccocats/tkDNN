#ifndef UTILSNN_H
#define UTILSNN_H

#include "tkdnn.h"
#include <opencv2/core/core.hpp>


void resizeAndSplit(cv::Mat& frame, uint8_t** frame_d, int& frame_size, dnnType *input_d, tk::dnn::NetworkRT *netRT, const int bi=0, bool BGR=true);

#endif // UTILSNN_H