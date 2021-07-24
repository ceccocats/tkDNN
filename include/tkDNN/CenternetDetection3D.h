#ifndef CENTERNETDETECTION3D_H
#define CENTERNETDETECTION3D_H

#include "kernels.h"
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort

#include "DetectionNN3D.h"

#include "kernelsThrust.h"


namespace tk { namespace dnn { 

class CenternetDetection3D : public DetectionNN3D
{
private:
    tk::dnn::dataDim_t dim;
    tk::dnn::dataDim_t dim2;
    tk::dnn::dataDim_t dim_hm; 
    tk::dnn::dataDim_t dim_wh; 
    tk::dnn::dataDim_t dim_reg;
    tk::dnn::dataDim_t dim_dep;
    tk::dnn::dataDim_t dim_rot;
    tk::dnn::dataDim_t dim_dim;

    std::vector<cv::Mat> inputCalibs;
    float *topk_scores;
    int *topk_inds_;
    float *topk_ys_;
    float *topk_xs_;
    int *ids_d, *ids_;

    float *ones;

    float *scores, *scores_d;
    int *clses, *clses_d; 
    int *topk_inds_d;
    float *topk_ys_d; 
    float *topk_xs_d;
    int *inttopk_xs_d, *inttopk_ys_d;

    float *xs, *ys;

    float *dep, *rot, *dim_, *wh;
    float *dep_d, *rot_d, *dim_d, *wh_d;
    
    float *target_coords;

    #ifdef OPENCV_CUDACONTRIB
        float *mean_d;
        float *stddev_d;
    #else
        cv::Vec<float, 3> mean;
        cv::Vec<float, 3> stddev;
        dnnType *input;
    #endif
    cv::Mat r;
    float *d_ptrs;
    
    cv::Size sz_old;

    cv::Mat src;
    cv::Mat dst;
    cv::Mat dst2;  
    cv::Mat trans, trans2;
    std::vector<cv::Mat> calibs;

    //processing
    int K = 100;
    int width = 128;//56;        // TODO

    // pointer used in the kernels
    float *srcOut;
    int *idsOut;

    struct threshold op;
    cv::Mat corners, pts3DHomo;  

    std::vector<std::vector<int>> faceId;

public:
    CenternetDetection3D() {};
    ~CenternetDetection3D() {}; 

    bool init(const std::string& tensor_path, const int n_classes=3, const int n_batches=1, const float conf_thresh=0.3, const std::vector<cv::Mat>& k_calibs=std::vector<cv::Mat>());
    void preprocess(cv::Mat &frame, const int bi=0);
    void postprocess(const int bi=0,const bool mAP=false);
    void draw(std::vector<cv::Mat>& frames);
};


} // namespace dnn
} // namespace tk


#endif /*CENTERNETDETECTION_H*/