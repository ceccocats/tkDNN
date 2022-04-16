#ifndef CENTERNETDETECTION_H
#define CENTERNETDETECTION_H

#include "kernels.h"
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort

#include "DetectionNN.h"

#include "kernelsThrust.h"


namespace tk { namespace dnn { 

class CenternetDetection : public DetectionNN
{
private:
    tk::dnn::dataDim_t dim;
    tk::dnn::dataDim_t dim2;
    tk::dnn::dataDim_t dim_hm; 
    tk::dnn::dataDim_t dim_wh; 
    tk::dnn::dataDim_t dim_reg;
    float *topk_scores;
    int *topk_inds_;
    float *topk_ys_;
    float *topk_xs_;
    int *ids_d, *ids_, *ids_2, *ids_2d;

    float *scores, *scores_d;
    int *clses, *clses_d; 
    int *topk_inds_d;
    float *topk_ys_d; 
    float *topk_xs_d;
    int *inttopk_xs_d, *inttopk_ys_d;


    float *bbx0, *bby0, *bbx1, *bby1;
    float *bbx0_d, *bby0_d, *bbx1_d, *bby1_d;
    
    float *target_coords;

    #ifdef OPENCV_CUDACONTRIB
        float *mean_d;
        float *stddev_d;
    #else
        cv::Vec<float, 3> mean;
        cv::Vec<float, 3> stddev;
        dnnType *input;
    #endif

    float *d_ptrs;

    cv::Mat src;
    cv::Mat dst;
    cv::Mat dst2;  
    cv::Mat trans, trans2;
    //processing
    float toll = 0.000001;
    int K = 100;
    int width = 128;//56;        // TODO

    // pointer used in the kernels
    float *src_out;
    int *ids_out;

    struct threshold op;
    
public:
    CenternetDetection() {};
    ~CenternetDetection() {}; 

    bool init(const std::string& tensor_path, const int n_classes=80, const int n_batches=1, const float conf_thresh=0.3);
    void preprocess(cv::Mat &frame, const int bi=0);
    void postprocess(const int bi=0,const bool mAP=false);
};


} // namespace dnn
} // namespace tk


#endif /*CENTERNETDETECTION_H*/